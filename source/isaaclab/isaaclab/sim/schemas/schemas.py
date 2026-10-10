# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import inspect
import logging
import math
from collections.abc import Callable, Iterable

import numpy as np
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from ...utils.string import to_camel_case
from ..utils import (
    create_prim,
    find_global_fixed_joint_prim,
    find_matching_prims,
    get_all_matching_child_prims,
    has_deformable_body_api,
    safe_set_attribute_on_usd_prim,
    safe_set_attribute_on_usd_schema,
)
from ..utils.stage import get_current_stage
from . import schemas_cfg
from .backend_hooks import skip_joint_drive

logger = logging.getLogger(__name__)


"""
Constants.
"""

# Mapping from string names to USD/PhysX tokens for mesh collision approximation
# Refer to omniverse documentation
# https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html#mesh-geometry-colliders
# for available tokens.
MESH_APPROXIMATION_TOKENS = {
    "boundingCube": UsdPhysics.Tokens.boundingCube,
    "boundingSphere": UsdPhysics.Tokens.boundingSphere,
    "convexDecomposition": UsdPhysics.Tokens.convexDecomposition,
    "convexHull": UsdPhysics.Tokens.convexHull,
    "none": UsdPhysics.Tokens.none,
    "meshSimplification": UsdPhysics.Tokens.meshSimplification,
    "sdf": "sdf",  # PhysX SDF mesh token; use string (pxr.Tf.Token not available in all envs)
}


"""
Schema-application helper.
"""


def _get_field_declaring_class(cfg_class: type, field_name: str) -> type | None:
    """Return the most-base class in the MRO that declares ``field_name``.

    Each cfg field is owned by a single class in the hierarchy (the one whose body
    contains its annotation). This function walks the MRO in reverse so a base class
    declaration wins over a subclass redeclaration with the same name -- the field's
    USD namespace follows where it semantically lives, not where it was last
    overridden for default values.
    """
    for cls in reversed(cfg_class.__mro__):
        if field_name in inspect.get_annotations(cls):
            return cls
    return None


def apply_namespaced_schemas(prim, cfg, cfg_dict: dict) -> None:
    """Route every cfg field to its declaring class's namespace and apply schemas.

    The helper handles the common ``AddAppliedSchema`` + namespaced-attribute write
    logic shared by every metadata-driven writer. Caller is responsible for popping
    fields that need typed-API writes (multi-instance ``UsdPhysics.DriveAPI``,
    ``TfToken`` attributes with ``allowedTokens``) out of ``cfg_dict`` first.

    USD attribute names are derived by snake_case -> camelCase conversion of cfg field
    names. The codebase enforces this as a convention: any cfg field whose
    snake_case name does not produce the correct USD camelCase attr is renamed (with a
    deprecation alias forwarded in ``__post_init__``) rather than mapped via metadata.

    Two passes:

    1. **Per-field exceptions** -- ``cfg._usd_field_exceptions`` is a mapping
       ``applied_schema -> (namespace, [cfg_field, ...])``. For each schema, if any
       listed field is non-None, the schema is applied (once) and each non-None field is
       written under that schema's namespace. Fields are popped from ``cfg_dict``.
    2. **Per-declaring-class routing** -- each remaining non-None field is grouped by the
       class that declares it (walking the MRO). Each group writes under that class's
       ``_usd_namespace`` and applies that class's ``_usd_applied_schema`` (if any). This
       means base-class fields go under the base namespace (e.g. ``physics:*``) even when
       the cfg instance is a backend subclass -- the subclass's ``_usd_namespace`` (e.g.
       ``"physxMaterial"``) only governs *its own* fields.

    Args:
        prim: The USD prim to author on.
        cfg: The cfg instance carrying the metadata.
        cfg_dict: A mutable dict view of the cfg's non-metadata fields. Modified in place.

    Raises:
        ValueError: If a non-None field's declaring class does not define ``_usd_namespace``.
    """
    cfg_class = type(cfg)

    # 1. Per-field exceptions (override per-class routing for fields whose USD home is another
    #    schema's namespace).
    field_exceptions = getattr(cfg, "_usd_field_exceptions", {}) or {}
    for applied_schema, (exc_ns, fields) in field_exceptions.items():
        triggered: list[tuple[str, object]] = []
        for cfg_field in fields:
            if cfg_field in cfg_dict:
                value = cfg_dict.pop(cfg_field)
                if value is not None:
                    triggered.append((to_camel_case(cfg_field, "cC"), value))
        if not triggered:
            continue
        if applied_schema and applied_schema not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema(applied_schema)
        for usd_attr, value in triggered:
            safe_set_attribute_on_usd_prim(prim, f"{exc_ns}:{usd_attr}", value, camel_case=False)

    # 2. Group remaining non-None writes by declaring class.
    by_class: dict[type, list[tuple[str, object]]] = {}
    for cfg_field, value in cfg_dict.items():
        if value is None:
            continue
        decl_class = _get_field_declaring_class(cfg_class, cfg_field)
        if decl_class is None:
            continue
        by_class.setdefault(decl_class, []).append((to_camel_case(cfg_field, "cC"), value))

    for decl_class, writes in by_class.items():
        # Read namespace/schema from the declaring class's own ``__dict__`` (not via
        # ``getattr``) so subclass overrides don't leak into base-field routing.
        namespace = decl_class.__dict__.get("_usd_namespace", None)
        applied_schema = decl_class.__dict__.get("_usd_applied_schema", None)
        if namespace is None:
            raise ValueError(
                f"{decl_class.__name__} declares fields {[a for a, _ in writes]} but does"
                " not define '_usd_namespace'. Add '_usd_namespace' to the class metadata"
                " or route the fields via '_usd_field_exceptions'."
            )
        if applied_schema and applied_schema not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema(applied_schema)
        for usd_attr, value in writes:
            safe_set_attribute_on_usd_prim(prim, f"{namespace}:{usd_attr}", value, camel_case=False)


def apply_namespaced(cfg: schemas_cfg.SchemaFragment, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Default fragment applier: apply the fragment's schema and write its namespaced attrs.

    Reads :attr:`~isaaclab.sim.schemas.SchemaFragment._usd_namespace` /
    :attr:`~isaaclab.sim.schemas.SchemaFragment._usd_applied_schema` from the cfg's class. If the
    fragment owns an applied schema, it is applied (once). Each non-``None`` dataclass field is
    written as ``<namespace>:<camelCase(field)>``; the ``func`` field is skipped. ``None`` fields
    are left unchanged on the prim (partial update).

    Args:
        cfg: The fragment instance carrying ``_usd_namespace`` / ``_usd_applied_schema`` metadata.
        prim_path: The prim path to author on.
        stage: The stage where to find the prim. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if the properties were successfully set.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    namespace = type(cfg)._usd_namespace
    applied = type(cfg)._usd_applied_schema
    if namespace is None:
        raise ValueError(
            f"Fragment '{type(cfg).__name__}' has no '_usd_namespace' set. Every fragment field is"
            " authored as '<namespace>:<attr>', so a USD namespace is required; non-USD state must"
            " live on the spawner cfg or be passed as a writer keyword argument, not as a fragment"
            " field."
        )
    if applied and applied not in prim.GetAppliedSchemas():
        prim.AddAppliedSchema(applied)
    # ``func`` is the only non-USD field; ``mesh_approximation_name`` is the shared ``physics:approximation``
    # token written by ``apply_mesh_collision``, not a ``<namespace>:meshApproximationName`` attribute
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        if f.name in ("func", "mesh_approximation_name") or value is None:
            continue
        safe_set_attribute_on_usd_prim(prim, f"{namespace}:{to_camel_case(f.name, 'cC')}", value, camel_case=False)
    return True


"""
Articulation root properties.
"""


def apply_articulation_root_properties(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.ArticulationRootFragment],
    stage: Usd.Stage | None = None,
    fix_root_link: bool | None = None,
    create_if_missing: bool = False,
) -> bool:
    """Apply a list of articulation-root fragments to the roots matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. Matched prims that
    already carry ``UsdPhysics.ArticulationRootAPI`` are the targets: each fragment is
    dispatched to every target via its :attr:`~isaaclab.sim.schemas.SchemaFragment.func`.
    Sibling roots (independent articulations matched by one expression) are all processed.
    Nested targets are authored as matched, with a warning -- resolving nested roots is the
    asset author's responsibility.

    With :paramref:`create_if_missing`, the API is applied to every matched prim that lacks
    it. Zero targets warn and return False. Instanced matches are skipped with a warning.

    An empty fragment list is an authoring no-op: it returns True immediately when
    :paramref:`fix_root_link` is None, but still resolves targets and adjusts topology when the
    flag is set. When :paramref:`fix_root_link` is True, the active physics manager creates or
    enables the world joint on each target and returns the backend's final root prim; False
    only disables an existing joint.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: Articulation-root fragments to apply.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.
        fix_root_link: Whether to fix the root link. None leaves topology unchanged.
        create_if_missing: Whether to apply ``UsdPhysics.ArticulationRootAPI`` to every
            matched prim that does not carry it. Defaults to False.

    Returns:
        True if every target and fragment succeeded and no instanced prim was skipped.

    Raises:
        TypeError: If fragments contains a non-articulation fragment.
        RuntimeError: If fixing cannot resolve the active backend or relocate the root.
        NotImplementedError: If the backend cannot fix the resolved root.
    """
    fragments = list(fragments)
    for fragment in fragments:
        if not isinstance(fragment, schemas_cfg.ArticulationRootFragment):
            raise TypeError(f"Expected ArticulationRootFragment, got '{type(fragment).__name__}'.")
    if stage is None:
        stage = get_current_stage()
    if not fragments and fix_root_link is None:
        return True

    targets, creation_candidates, any_skipped = _match_fragment_targets(
        prim_path_expr, lambda p: p.HasAPI(UsdPhysics.ArticulationRootAPI), stage
    )
    if create_if_missing:
        for prim in creation_candidates:
            UsdPhysics.ArticulationRootAPI.Apply(prim)
            targets.append(prim)
    target_paths = [t.GetPath() for t in targets]
    if any(path != other and path.HasPrefix(other) for path in target_paths for other in target_paths):
        logger.warning(
            "Expression '%s' targets nested articulation roots (%s); authoring on all of them.",
            prim_path_expr,
            [p.pathString for p in target_paths],
        )
    if not targets:
        logger.warning("No articulation-root targets matched expression '%s'; nothing was authored.", prim_path_expr)
        return False

    if fix_root_link:
        from .. import SimulationContext

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError(
                f"Cannot fix articulation roots matched by '{prim_path_expr}' without an active simulation."
            )

    # aggregate per-target, per-fragment results so a reported failure is not masked
    success = not any_skipped
    for root in targets:
        if fix_root_link:
            root = sim.physics_manager.fix_articulation_root(root, stage)
        elif fix_root_link is False:
            joint = find_global_fixed_joint_prim(root.GetPath().pathString, stage=stage)
            if joint is not None:
                joint.GetJointEnabledAttr().Set(False)

        root_path = root.GetPath().pathString
        for fragment in fragments:
            success = bool(fragment.func(fragment, root_path, stage)) and success

    return success


def create_world_fixed_joint(articulation_prim: Usd.Prim, stage: Usd.Stage) -> None:
    """Author a ``UsdPhysics.FixedJoint`` fixing an articulation root link to the world frame.

    This is a pure-USD equivalent of
    ``omni.physx.scripts.utils.createJoint(joint_type="Fixed", from_prim=None, to_prim=articulation_prim)``.
    Authoring directly with USD keeps the fixed-root-link spawn path backend-agnostic:
    it works identically under Kit/PhysX and on kitless backends (e.g. Newton) where
    ``omni.physx`` is unavailable. Only the single-body (world-attached) case is handled,
    matching the fixed-root-link spawn path.

    Args:
        articulation_prim: The articulation root link prim to fix to the world.
        stage: The stage that owns the prim.
    """
    # ``MAX_FLOAT`` used by ``omni.physx.createJoint`` for an effectively unbreakable joint.
    max_break = 3.40282347e38

    to_path = articulation_prim.GetPath().pathString

    # Instanceable/prototype/instance-proxy prims are not authorable; walk up to the first
    # writable ancestor so the joint can be defined there (mirrors ``omni.physx.createJoint``).
    base_prim = articulation_prim
    pseudo_root = stage.GetPseudoRoot()
    while base_prim != pseudo_root and (
        base_prim.IsInPrototype() or base_prim.IsInstanceProxy() or base_prim.IsInstanceable()
    ):
        base_prim = base_prim.GetParent()
    joint_base_path = str(base_prim.GetPrimPath())
    if joint_base_path == "/":
        joint_base_path = ""

    # Find a unique joint name under the writable base (mirrors ``create_unused_path``).
    joint_name = "FixedJoint"
    if stage.GetPrimAtPath(f"{joint_base_path}/{joint_name}").IsValid():
        uniquifier = 0
        while stage.GetPrimAtPath(f"{joint_base_path}/{joint_name}{uniquifier}").IsValid():
            uniquifier += 1
        joint_name = f"{joint_name}{uniquifier}"
    joint = UsdPhysics.FixedJoint.Define(stage, f"{joint_base_path}/{joint_name}")

    # Anchor the joint at the root link's world pose (body0 = world, body1 = root link).
    world_pose = UsdGeom.XformCache().GetLocalToWorldTransform(articulation_prim).RemoveScaleShear()
    joint.CreateBody1Rel().SetTargets([Sdf.Path(to_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(world_pose.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(world_pose.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.CreateBreakForceAttr().Set(max_break)
    joint.CreateBreakTorqueAttr().Set(max_break)


"""
Fragment-writer helpers.
"""


def _match_fragment_targets(
    prim_path_expr: str,
    is_target: Callable[[Usd.Prim], bool],
    stage: Usd.Stage,
) -> tuple[list[Usd.Prim], list[Usd.Prim], bool]:
    """Resolve fragment-writer targets from a prim path expression.

    Matches ``prim_path_expr`` with :func:`~isaaclab.sim.utils.find_matching_prims` (a plain
    regular expression over whole prim paths) and splits the matches: writable prims
    passing ``is_target`` are targets, writable prims failing it are creation candidates, and
    instanced prims passing it are skipped with a warning since prototypes cannot be authored on.

    Args:
        prim_path_expr: The prim path expression to match. Path-like objects (e.g.
            ``Sdf.Path``) are accepted and converted with ``str``.
        is_target: Predicate deciding whether a matched prim is a valid family target.
        stage: The stage to match on.

    Returns:
        A tuple ``(targets, creation_candidates, any_skipped)``.
    """
    # tolerate path-like inputs such as Sdf.Path, whose string form is the path itself
    prim_path_expr = str(prim_path_expr)
    targets = []
    creation_candidates = []
    skipped = []
    for prim in find_matching_prims(prim_path_expr, stage):
        instanced = prim.IsInstance() or prim.IsInstanceProxy()
        if is_target(prim):
            (skipped if instanced else targets).append(prim)
        elif not instanced:
            creation_candidates.append(prim)
    if skipped:
        logger.warning(
            "Skipping fragment updates on instanced prims matched by '%s': %s.",
            prim_path_expr,
            [p.GetPath().pathString for p in skipped],
        )
    return targets, creation_candidates, bool(skipped)


def _apply_api_fragments(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.SchemaFragment],
    api: type[Usd.APISchemaBase],
    family: str,
    create_if_missing: bool,
    stage: Usd.Stage | None,
) -> bool:
    """Dispatch fragments to every matched prim carrying ``api``, the family's implicit anchor.

    Shared by the rigid-body, collision and mass writers: an empty fragment list is a no-op that
    returns True, ``create_if_missing`` applies ``api`` to matched prims lacking it, zero targets
    warn and return False, and per-target results are aggregated so a failure is never masked.
    """
    fragments = list(fragments)
    if stage is None:
        stage = get_current_stage()
    if not fragments:
        return True
    targets, creation_candidates, any_skipped = _match_fragment_targets(prim_path_expr, lambda p: p.HasAPI(api), stage)
    if create_if_missing:
        for prim in creation_candidates:
            api.Apply(prim)
            targets.append(prim)
    if not targets:
        logger.warning("No %s targets matched expression '%s'; nothing was authored.", family, prim_path_expr)
        return False
    success = not any_skipped
    for target in targets:
        target_path = target.GetPath().pathString
        for cfg in fragments:
            success = bool(cfg.func(cfg, target_path, stage)) and success
    return success


"""
Rigid body properties.
"""


def apply_rigid_body_properties(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.RigidBodyFragment],
    create_if_missing: bool = False,
    stage: Usd.Stage | None = None,
) -> bool:
    """Apply a list of rigid-body fragments to the rigid bodies matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. Matched prims that
    already carry ``UsdPhysics.RigidBodyAPI`` are modified in place: each fragment is
    dispatched to every such target via its
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func`. Backend fragments carry backend-specific
    funcs, so core never imports a backend.

    An empty fragment list is an authoring no-op and returns True. With
    :paramref:`create_if_missing`, ``UsdPhysics.RigidBodyAPI`` is applied to every matched
    prim that lacks it; only the asset's joints decide which bodies participate in the
    articulation, so the expression is trusted as written. Zero targets warn and return
    False. Instanced matches are skipped with a warning.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.RigidBodyFragment` instances.
        create_if_missing: Whether to apply ``UsdPhysics.RigidBodyAPI`` to every matched
            prim that does not carry it. Defaults to False.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if every target and fragment succeeded and no instanced prim was skipped.
    """
    return _apply_api_fragments(
        prim_path_expr, fragments, UsdPhysics.RigidBodyAPI, "rigid-body", create_if_missing, stage
    )


"""
Deformable body properties.
"""


# Simulation-mesh API schemas that record whether an authored deformable body is a volume or a
# surface deformable. The generic deformable-body anchor on the body prim does not carry that
# split, so both backend spellings are matched here.
_DEFORMABLE_SIM_API_TYPES = {
    "OmniPhysicsVolumeDeformableSimAPI": "volume",
    "PhysicsVolumeDeformableSimAPI": "volume",
    "OmniPhysicsSurfaceDeformableSimAPI": "surface",
    "PhysicsSurfaceDeformableSimAPI": "surface",
}


def _authored_deformable_type(prim: Usd.Prim) -> str | None:
    """Read this body’s simulation type, excluding nested deformable bodies."""
    authored_types = set()
    descendants = iter(Usd.PrimRange(prim))
    for descendant in descendants:
        if descendant != prim and has_deformable_body_api(descendant):
            descendants.PruneChildren()
            continue
        for schema in descendant.GetPrimTypeInfo().GetAppliedAPISchemas():
            authored_type = _DEFORMABLE_SIM_API_TYPES.get(schema)
            if authored_type is not None:
                authored_types.add(authored_type)
    return authored_types.pop() if len(authored_types) == 1 else None


def _apply_deformable_body_properties(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.DeformableBodyFragment],
    deformable_type: str,
    create_if_missing: bool,
    stage: Usd.Stage | None,
    sim_mesh_name: str,
    tetrahedralization_edge_length_fac: float,
) -> bool:
    """Shared implementation of the volume/surface deformable family writers."""
    fragments = list(fragments)
    if stage is None:
        stage = get_current_stage()
    if not fragments and not create_if_missing:
        return True
    for cfg in fragments:
        if deformable_type not in type(cfg)._deformable_types:
            logger.warning(
                "Fragment '%s' is not meaningful for %s deformables; authoring anyway.",
                type(cfg).__name__,
                deformable_type,
            )
    targets, creation_candidates, any_skipped = _match_fragment_targets(prim_path_expr, has_deformable_body_api, stage)
    # the deformable-body anchor is type-agnostic, so a prim authored as the other deformable type
    # also matches here. Authoring this family onto it would leave the body simulating as its
    # authored type while carrying this family's attributes, so drop it instead.
    matched_targets, targets = targets, []
    for prim in matched_targets:
        authored_type = _authored_deformable_type(prim)
        if authored_type is not None and authored_type != deformable_type:
            any_skipped = True
            logger.warning(
                "Prim '%s' is already authored as a %s deformable but was matched by '%s' as a %s"
                " deformable; skipping it. Author it through the %s deformable family instead.",
                prim.GetPath().pathString,
                authored_type,
                prim_path_expr,
                deformable_type,
                authored_type,
            )
        else:
            targets.append(prim)
    if create_if_missing and creation_candidates:
        # Keep this import local to avoid the SimulationContext -> schemas import cycle.
        from isaaclab.sim import SimulationContext  # noqa: PLC0415

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError(
                f"Cannot create deformable bodies matched by '{prim_path_expr}' without an active simulation."
            )
        for prim in creation_candidates:
            sim_mesh_prim_path = f"{prim.GetPath().pathString}/{sim_mesh_name}"
            sim_mesh_prim, vis_mesh_prim = _setup_deformable_meshes(
                prim, deformable_type, sim_mesh_prim_path, stage, tetrahedralization_edge_length_fac
            )
            sim.physics_manager.setup_deformable_body(prim, deformable_type, sim_mesh_prim, vis_mesh_prim)
            targets.append(prim)
    if not targets:
        if not any_skipped:
            logger.warning("No deformable-body targets matched expression '%s'; nothing was authored.", prim_path_expr)
        return False
    # aggregate per-target, per-fragment results so a reported failure is not masked
    success = not any_skipped
    for target in targets:
        target_path = target.GetPath().pathString
        for cfg in fragments:
            success = bool(cfg.func(cfg, target_path, stage)) and success
    return success


def apply_volume_deformable_properties(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.DeformableBodyFragment],
    create_if_missing: bool = False,
    stage: Usd.Stage | None = None,
    sim_mesh_name: str = "sim_mesh",
    tetrahedralization_edge_length_fac: float = 0.1,
) -> bool:
    """Apply deformable-body fragments to the volume deformables matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`,
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Body/.*`` every descendant of a prim. Matched prims that
    already carry a deformable-body anchor (per :func:`~isaaclab.sim.utils.has_deformable_body_api`)
    are modified in place: each fragment is dispatched to every such target via its
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func`.

    With :paramref:`create_if_missing`, every matched prim without the anchor receives a full
    volume-deformable setup: the simulation ``TetMesh`` is created as a ``sim_mesh_name`` child
    (reusing a pre-tetrahedralized ``UsdGeom.TetMesh`` child when present, tetrahedralizing the
    visual mesh via the optional ``pytetwild`` package otherwise), collision is enabled on it,
    and the anchor schemas are applied through the active physics backend. Creation therefore
    requires an active simulation. Zero targets warn and return False; instanced matches are
    skipped with a warning; fragments not meaningful for volume deformables warn but author.
    Matched prims already authored as surface deformables are skipped with a warning rather than
    being given a volume family they do not simulate with.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.DeformableBodyFragment` instances.
        create_if_missing: Whether to run the full deformable setup on matched prims that do
            not carry the anchor. Defaults to False.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.
        sim_mesh_name: Name of the simulation-mesh child prim created per target. Defaults to
            ``"sim_mesh"``.
        tetrahedralization_edge_length_fac: Relative target edge length for automatic
            tetrahedralization. Defaults to ``0.1``.

    Returns:
        True if every target and fragment succeeded and no matched prim was skipped.

    Raises:
        RuntimeError: If creation is requested without an active simulation.
    """
    return _apply_deformable_body_properties(
        prim_path_expr, fragments, "volume", create_if_missing, stage, sim_mesh_name, tetrahedralization_edge_length_fac
    )


def apply_surface_deformable_properties(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.DeformableBodyFragment],
    create_if_missing: bool = False,
    stage: Usd.Stage | None = None,
    sim_mesh_name: str = "sim_mesh",
) -> bool:
    """Apply deformable-body fragments to the surface deformables matched by an expression.

    Same contract as :func:`apply_volume_deformable_properties` with surface structural work:
    the simulation mesh is a triangle-mesh copy of the visual mesh (no tetrahedralization), and
    matched prims already authored as volume deformables are the ones skipped with a warning.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.DeformableBodyFragment` instances.
        create_if_missing: Whether to run the full deformable setup on matched prims that do
            not carry the anchor. Defaults to False.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.
        sim_mesh_name: Name of the simulation-mesh child prim created per target. Defaults to
            ``"sim_mesh"``.

    Returns:
        True if every target and fragment succeeded and no matched prim was skipped.

    Raises:
        RuntimeError: If creation is requested without an active simulation.
    """
    return _apply_deformable_body_properties(
        prim_path_expr, fragments, "surface", create_if_missing, stage, sim_mesh_name, 0.1
    )


def apply_mesh_collision(
    cfg: schemas_cfg.MeshCollisionFragment, prim_path: str, stage: Usd.Stage | None = None
) -> bool:
    """Apply a single mesh-collision fragment: its namespaced cooking attrs plus the shared token.

    This is the default :attr:`~isaaclab.sim.schemas.SchemaFragment.func` for every
    :class:`~isaaclab.sim.schemas.MeshCollisionFragment`. Unlike the generic :func:`apply_namespaced`,
    a mesh-collision fragment additionally authors the shared ``physics:approximation`` token (via the
    standard ``UsdPhysics.MeshCollisionAPI``) on top of its own backend cooking namespace.

    The token is *not* a plain namespaced attribute -- it is shared state on the family anchor implied
    by the present cooking fragment. The core and PhysX fragments carry a :attr:`mesh_approximation_name`
    whose default encodes the token their schema implies (e.g. ``"convexHull"`` for :class:`PhysxConvexHullCfg`,
    ``"sdf"`` for :class:`PhysxSDFMeshCfg`); the Newton cooking fragments carry none and leave the token
    unchanged. A name of ``"none"`` also leaves the token unchanged, so when
    several fragments are dispatched in order by :func:`apply_mesh_collision_properties` the last one
    with a non-``"none"`` name wins -- this is how a core fragment composes with a backend cooking
    fragment. The name is validated against :const:`MESH_APPROXIMATION_TOKENS`; an unknown name raises
    ``ValueError``. :attr:`mesh_approximation_name` is skipped by :func:`apply_namespaced`, so it is
    never authored as a spurious ``<namespace>:meshApproximationName`` attribute.

    Args:
        cfg: The mesh-collision fragment to apply.
        prim_path: The prim path to author on. This prim should be a Mesh.
        stage: The stage where to find the prim. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if the fragment was applied successfully.

    Raises:
        ValueError: If the prim at ``prim_path`` is not valid, or when the fragment's mesh
            approximation name is not in :const:`MESH_APPROXIMATION_TOKENS`.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # the standard MeshCollisionAPI anchor carries ``physics:approximation``
    if not UsdPhysics.MeshCollisionAPI(prim):
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    # the generic applier skips ``mesh_approximation_name``; the shared token is written below
    success = apply_namespaced(cfg, prim_path, stage)
    # ``"none"`` leaves the token untouched so a later non-"none" fragment in a list dispatch wins
    name = getattr(cfg, "mesh_approximation_name", None)
    if name is not None and name != "none":
        _write_mesh_approximation(prim, name)
    return success


def _write_mesh_approximation(prim: Usd.Prim, name: str) -> None:
    """Author ``physics:approximation`` on a ``MeshCollisionAPI`` prim from a token name."""
    if name not in MESH_APPROXIMATION_TOKENS:
        raise ValueError(
            f"Invalid mesh approximation name: '{name}'. Valid options are: {list(MESH_APPROXIMATION_TOKENS)}"
        )
    safe_set_attribute_on_usd_schema(
        UsdPhysics.MeshCollisionAPI(prim), "Approximation", MESH_APPROXIMATION_TOKENS[name], camel_case=False
    )


def apply_mesh_collision_properties(
    prim_path: str, fragments: Iterable[schemas_cfg.MeshCollisionFragment], stage: Usd.Stage | None = None
) -> bool:
    """Apply a list of mesh-collision fragments to a prim.

    Applies ``UsdPhysics.MeshCollisionAPI`` as the implicit anchor (the carrier of the
    ``physics:approximation`` token), then dispatches each fragment via its
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func`. The default mesh-collision func
    (:func:`apply_mesh_collision`) authors both the fragment's backend cooking namespace and the
    shared approximation token it implies, so composing a core fragment with a backend cooking
    fragment lets the last fragment with a non-``"none"`` :attr:`mesh_approximation_name` set the
    token. Backend cooking fragments carry their own funcs, so core never imports a backend.

    Args:
        prim_path: The prim path to apply the mesh-collision schemas on. This prim should be a Mesh.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.MeshCollisionFragment` instances.
        stage: The stage where to find the prim. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if all fragments applied successfully, False if any fragment reported failure.

    Raises:
        ValueError: If the prim at ``prim_path`` is not valid, or when a fragment's mesh
            approximation name is not in :const:`MESH_APPROXIMATION_TOKENS`.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    if not UsdPhysics.MeshCollisionAPI(prim):
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    # aggregate per-fragment results so a reported failure is not masked
    success = True
    for cfg in fragments:
        success = bool(cfg.func(cfg, prim_path, stage)) and success
    return success


"""
Collision properties.
"""


def apply_collision_properties(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.CollisionFragment],
    create_if_missing: bool = False,
    stage: Usd.Stage | None = None,
) -> bool:
    """Apply a list of collision fragments to the colliders matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. Matched prims that
    already carry ``UsdPhysics.CollisionAPI`` are modified in place: each fragment is
    dispatched to every such target via its
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func`. Backend fragments carry backend-specific
    funcs, so core never imports a backend.

    An empty fragment list is an authoring no-op and returns True. With
    :paramref:`create_if_missing`, ``UsdPhysics.CollisionAPI`` is applied to every matched
    prim that lacks it. When no target remains, a warning is emitted and False is
    returned without authoring anything. Matched prims inside instances cannot be authored on
    and are skipped with a warning.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.CollisionFragment` instances.
        create_if_missing: Whether to apply ``UsdPhysics.CollisionAPI`` to matched prims that
            do not carry it. Defaults to False.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if every target and fragment succeeded and no instanced prim was skipped.
    """
    return _apply_api_fragments(
        prim_path_expr, fragments, UsdPhysics.CollisionAPI, "collision", create_if_missing, stage
    )


"""
Mass properties.
"""


def apply_mass_properties(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.MassFragment],
    create_if_missing: bool = False,
    stage: Usd.Stage | None = None,
) -> bool:
    """Apply a list of mass fragments to the mass-bearing prims matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. Matched prims that
    already carry ``UsdPhysics.MassAPI`` are modified in place: each fragment is dispatched to
    every such target via its :attr:`~isaaclab.sim.schemas.SchemaFragment.func`. Backend
    fragments carry backend-specific funcs, so core never imports a backend.

    An empty fragment list is an authoring no-op and returns True. With
    :paramref:`create_if_missing`, ``UsdPhysics.MassAPI`` is applied to every matched prim
    that lacks it; pairing the mass with a rigid body is the caller's responsibility. Zero
    targets warn and return False. Instanced matches are skipped with a warning.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.MassFragment` instances.
        create_if_missing: Whether to apply ``UsdPhysics.MassAPI`` to every matched prim
            that does not carry it. Defaults to False.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if every target and fragment succeeded and no instanced prim was skipped.
    """
    return _apply_api_fragments(prim_path_expr, fragments, UsdPhysics.MassAPI, "mass", create_if_missing, stage)


"""
Contact sensor.
"""


def activate_contact_sensors(prim_path: str, threshold: float = 0.0, stage: Usd.Stage | None = None):
    """Activate the contact sensor on all rigid bodies under a specified prim path.

    This function adds the PhysX contact report API to all rigid bodies under the specified prim path.
    It also sets the force threshold beyond which the contact sensor reports the contact. The contact
    reporting API can only be added to rigid bodies.

    Args:
        prim_path: The prim path under which to search and prepare contact sensors.
        threshold: The threshold for the contact sensor. Defaults to 0.0.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: If the input prim path is not valid.
        ValueError: If there are no rigid bodies under the prim path.
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # nested rigid-body trees are included
    rigid_body_prims = get_all_matching_child_prims(
        prim_path,
        predicate=lambda child_prim: child_prim.HasAPI(UsdPhysics.RigidBodyAPI),
        stage=stage,
        traverse_instance_prims=False,
    )
    if not rigid_body_prims:
        descendant_count = sum(1 for _ in Usd.PrimRange(prim)) - 1
        logger.warning(
            "[activate_contact_sensors] no rigid bodies found under prim=%r (type=%r, descendants=%d)",
            prim_path,
            prim.GetTypeName(),
            descendant_count,
        )
        raise ValueError(
            f"No contact sensors added to the prim: '{prim_path}'. This means that no rigid bodies"
            " are present under this prim. Please check the prim path."
        )
    for child_prim in rigid_body_prims:
        child_applied = child_prim.GetAppliedSchemas()
        # a zero sleep threshold keeps the body awake so contacts are always reported
        if "PhysxRigidBodyAPI" not in child_applied:
            child_prim.AddAppliedSchema("PhysxRigidBodyAPI")
        safe_set_attribute_on_usd_prim(child_prim, "physxRigidBody:sleepThreshold", 0.0, camel_case=False)
        if "PhysxContactReportAPI" not in child_applied:
            child_prim.AddAppliedSchema("PhysxContactReportAPI")
        safe_set_attribute_on_usd_prim(child_prim, "physxContactReport:threshold", threshold, camel_case=False)
    return True


"""
Joint drive properties.
"""


def drive_instance_name(prim) -> str | None:
    """Return the ``UsdPhysics.DriveAPI`` instance for a joint prim, or ``None`` if it has no drive.

    Revolute joints use the ``"angular"`` instance, prismatic joints the ``"linear"`` instance; any
    other prim type has no joint drive. Shared by :func:`apply_drive` and :func:`_ensure_drive_exists`.

    Args:
        prim: The candidate joint prim.

    Returns:
        ``"angular"``, ``"linear"``, or ``None`` when the prim is not a revolute/prismatic joint.
    """
    if prim.IsA(UsdPhysics.RevoluteJoint):
        return "angular"
    if prim.IsA(UsdPhysics.PrismaticJoint):
        return "linear"
    return None


def apply_drive(cfg, prim_path: str, stage: Usd.Stage | None = None) -> bool:
    """Apply a :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` fragment to a single joint prim.

    This is the override ``func`` for the ``UsdPhysics.DriveAPI`` fragment: the drive attributes
    live under a multi-instance schema, so the generic :func:`apply_namespaced` writer cannot be
    used. The writer:

    * Selects the drive instance: ``"angular"`` for a revolute joint, ``"linear"`` for a prismatic
      joint. For any other prim type, the function is a no-op and returns ``False``.
    * Skips joints excluded by a backend-registered predicate (see
      :func:`register_joint_drive_skip_predicate`, e.g. PhysX tendon members), returning ``False``.
    * Applies ``UsdPhysics.DriveAPI`` for the selected instance (presence-gated -- only applied when
      this fragment is present).
    * Converts angular-drive :attr:`stiffness` and :attr:`damping` from radians to degrees
      (``N·m/rad`` -> ``N·m/deg`` and ``N·m·s/rad`` -> ``N·m·s/deg``); linear drives are written
      as-is.
    * Writes the typed ``drive:<inst>:physics:{type,maxForce,stiffness,damping}`` attributes,
      mapping the :attr:`drive_type` field to the USD attribute named ``type``.

    Args:
        cfg: The :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` fragment to apply.
        prim_path: The joint prim path to author on.
        stage: The stage where to find the prim. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if the drive was applied to a joint prim, False if the prim is not a revolute or
        prismatic joint (or is a tendon child).
    """
    if stage is None:
        stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    drive_api_name = drive_instance_name(prim)
    # skip non-joints and joints a backend owns (e.g. PhysX tendon members)
    if drive_api_name is None or skip_joint_drive(prim):
        return False
    usd_drive_api = UsdPhysics.DriveAPI(prim, drive_api_name) or UsdPhysics.DriveAPI.Apply(prim, drive_api_name)
    _write_drive_attributes(usd_drive_api, drive_api_name, cfg.drive_type, cfg.max_force, cfg.stiffness, cfg.damping)
    return True


def _write_drive_attributes(
    usd_drive_api: UsdPhysics.DriveAPI,
    drive_api_name: str,
    drive_type: str | None,
    max_force: float | None,
    stiffness: float | None,
    damping: float | None,
) -> None:
    """Write the solver-common ``UsdPhysics.DriveAPI`` attributes, skipping ``None`` values.

    Angular drives are stored in degree units in USD, so stiffness [N·m/rad] and damping
    [N·m·s/rad] are converted to per-degree values. ``drive_type`` maps to the USD attribute
    ``type``; every other field follows the snake_case -> camelCase convention.
    """
    if drive_api_name == "angular":
        if stiffness is not None:
            stiffness = stiffness * math.pi / 180.0
        if damping is not None:
            damping = damping * math.pi / 180.0
    for attr_name, value in (
        ("type", drive_type),
        ("max_force", max_force),
        ("stiffness", stiffness),
        ("damping", damping),
    ):
        if value is not None:
            safe_set_attribute_on_usd_schema(usd_drive_api, attr_name, value, camel_case=True)


def apply_joint_drive_properties(
    prim_path_expr: str,
    fragments,
    stage: Usd.Stage | None = None,
    ensure_drives_exist: bool = False,
    create_if_missing: bool = False,
) -> bool:
    """Apply a list of joint-drive fragments to the joint prims matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. The fragments are
    dispatched to every matched revolute/prismatic joint prim that is not excluded by a
    backend-registered skip predicate (see :func:`register_joint_drive_skip_predicate`, e.g.
    PhysX tendon members). Non-joint matches are ignored silently -- a subtree expression
    matches every descendant, so per-prim warnings would spam. Matched prims inside
    instances cannot be authored on and are skipped with a warning.

    Unlike :func:`apply_rigid_body_properties`, the joint-drive family has no implicit anchor:
    ``UsdPhysics.DriveAPI`` is *presence-gated* and applied only by :func:`apply_drive` when a
    :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` fragment is present in ``fragments``. Each
    fragment is dispatched via its :attr:`~isaaclab.sim.schemas.SchemaFragment.func`, so backend
    fragments carry backend-specific funcs and core never imports a backend.

    An empty fragment list is an authoring no-op and returns True. When no fragment succeeds on
    any joint, a warning is emitted and False is returned.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.JointDriveFragment` instances.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.
        ensure_drives_exist: If True, write a minimal stiffness (``1e-3``) to any drive whose
            authored stiffness *and* damping are both zero, so that backends (e.g. Newton) treat
            the drive as active. This is a spawner-level flag, not a fragment field.
        create_if_missing: If True, apply the axis-appropriate ``UsdPhysics.DriveAPI`` instance
            (``"angular"`` for revolute joints, ``"linear"`` for prismatic joints) on matched
            joints that do not carry it, before dispatching the fragments. Distinct from
            :paramref:`ensure_drives_exist`: this flag creates the drive API itself, whereas
            :paramref:`ensure_drives_exist` seeds a minimal stiffness on fully-passive drives
            that already exist. Defaults to False.

    Returns:
        True if the fragments were applied to at least one joint prim and no instanced joint
        was skipped, False otherwise.
    """
    if stage is None:
        stage = get_current_stage()
    fragments = list(fragments)
    if not fragments:
        return True
    # ``ensure_drives_exist`` only makes sense for the solver-common drive fragment
    drive_cfg = next((f for f in fragments if isinstance(f, schemas_cfg.UsdPhysicsDriveCfg)), None)

    # non-joint matches are ignored silently since a subtree expression matches every descendant
    targets, _, any_skipped = _match_fragment_targets(
        prim_path_expr, lambda p: drive_instance_name(p) is not None and not skip_joint_drive(p), stage
    )

    count_success = 0
    for joint_prim in targets:
        joint_prim_path = joint_prim.GetPath().pathString
        drive_api_name = drive_instance_name(joint_prim)
        if create_if_missing:
            if not UsdPhysics.DriveAPI(joint_prim, drive_api_name):
                UsdPhysics.DriveAPI.Apply(joint_prim, drive_api_name)
        results = [bool(cfg.func(cfg, joint_prim_path, stage)) for cfg in fragments]
        if not any(results):
            continue
        count_success += 1
        if ensure_drives_exist and drive_cfg is not None:
            _ensure_drive_exists(drive_cfg, joint_prim, drive_api_name)

    # instanced skips were already reported by the matcher; only warn when nothing matched at all
    if count_success == 0 and not any_skipped:
        logger.warning(
            "Could not apply joint-drive properties on any joints matched by '%s'."
            " No revolute/prismatic joint prims matched or every fragment reported failure.",
            prim_path_expr,
        )
    return count_success > 0 and not any_skipped


def _ensure_drive_exists(drive_cfg: schemas_cfg.UsdPhysicsDriveCfg, prim: Usd.Prim, drive_api_name: str) -> None:
    """Seed a minimal stiffness on a fully-passive drive so backends treat it as active.

    If the drive fragment authored neither :attr:`stiffness` nor :attr:`damping` and the authored
    drive currently has zero (or unset) stiffness *and* damping, write a minimal stiffness of
    ``1e-3`` directly to the drive API (converted to degree units for angular drives, matching
    :func:`apply_drive`). The fragment is not mutated, so this is safe across multiple joint prims
    sharing one fragment.

    Args:
        drive_cfg: The :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` fragment.
        prim: The joint prim being authored.
        drive_api_name: The drive instance of the joint (``"angular"`` or ``"linear"``).
    """
    if drive_cfg.stiffness is not None or drive_cfg.damping is not None:
        return
    usd_drive_api = UsdPhysics.DriveAPI(prim, drive_api_name) or UsdPhysics.DriveAPI.Apply(prim, drive_api_name)
    if not usd_drive_api.GetStiffnessAttr().Get() and not usd_drive_api.GetDampingAttr().Get():
        # 1e-3 is given in per-radian units, so an angular drive gets 1e-3 * pi / 180
        _write_drive_attributes(usd_drive_api, drive_api_name, None, None, 1e-3, None)


"""
Fixed tendon properties.
"""


_FIXED_TENDON_SCHEMAS = ("PhysxTendonAxisRootAPI", "PhysxTendonAxisAPI")
_SPATIAL_TENDON_SCHEMAS = ("PhysxTendonAttachmentRootAPI",)


def _apply_tendon_fragments(
    prim_path_expr: str,
    fragments: Iterable[schemas_cfg.SchemaFragment],
    schema_types: tuple[str, ...],
    prim_types: tuple[str, ...],
    family: str,
    stage: Usd.Stage | None,
) -> bool:
    """Dispatch tune-not-apply tendon fragments to prims carrying a tendon schema or prim type.

    A fragment succeeds when its func returns True on at least one target, so a mixed-backend
    target set (each func no-ops on the other backend's prims) does not fail the write.
    """
    fragments = list(fragments)
    if stage is None:
        stage = get_current_stage()
    if not fragments:
        return True
    targets, _, any_skipped = _match_fragment_targets(
        prim_path_expr,
        lambda prim: (
            prim.GetTypeName() in prim_types
            or any(
                Usd.SchemaRegistry.GetTypeNameAndInstance(str(schema))[0] in schema_types
                for schema in prim.GetAppliedSchemas()
            )
        ),
        stage,
    )
    if not targets:
        logger.warning("No %s-tendon targets matched expression '%s'; nothing was authored.", family, prim_path_expr)
        return False
    target_paths = [target.GetPath().pathString for target in targets]
    success = not any_skipped
    for cfg in fragments:
        # every target is visited; the list keeps ``any`` from short-circuiting the dispatch
        results = [bool(cfg.func(cfg, path, stage)) for path in target_paths]
        success = any(results) and success
    return success


def apply_fixed_tendon_properties(
    prim_path_expr: str, fragments: Iterable[schemas_cfg.FixedTendonFragment], stage: Usd.Stage | None = None
) -> bool:
    """Apply a list of fixed-tendon fragments to the tendon prims matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. A matched prim is a
    fixed-tendon target when it carries an applied ``PhysxTendonAxisRootAPI`` or
    ``PhysxTendonAxisAPI`` instance, or is a ``MjcTendon`` prim.

    Fixed tendons are a *tune-not-apply* family: the tendon topology is authored in the source
    asset, so this writer never creates instances -- it only dispatches each fragment via its
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func` to every matched target. Backend
    fragments carry backend-specific funcs, so core never imports a backend. A fragment
    succeeds when its func returns True on at least one target: each func only tunes its own
    backend's representation and no-ops (returns False) on the other backend's prims, so a
    mixed-backend target set does not fail the write.

    An empty fragment list is an authoring no-op and returns True. When no target matches, a
    warning is emitted and False is returned without authoring anything. Matched prims inside
    instances cannot be authored on and are skipped with a warning.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.FixedTendonFragment` instances.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if every fragment tuned at least one target and no instanced prim was skipped.
    """
    return _apply_tendon_fragments(prim_path_expr, fragments, _FIXED_TENDON_SCHEMAS, ("MjcTendon",), "fixed", stage)


"""
Spatial tendon properties.
"""


def apply_spatial_tendon_properties(
    prim_path_expr: str, fragments: Iterable[schemas_cfg.SpatialTendonFragment], stage: Usd.Stage | None = None
) -> bool:
    """Apply a list of spatial-tendon fragments to the tendon prims matched by an expression.

    The prims to author on are matched with :func:`~isaaclab.sim.utils.find_matching_prims`:
    ``prim_path_expr`` is a plain regular expression over whole prim paths, so ``[^/]+``
    selects one path segment and ``/World/Robot/.*`` every descendant of a prim. A matched prim is a
    spatial-tendon target when it carries an applied ``PhysxTendonAttachmentRootAPI`` instance.

    Spatial tendons are a *tune-not-apply* family: the tendon topology is authored in the
    source asset, so this writer never creates instances -- it only dispatches each fragment
    via its :attr:`~isaaclab.sim.schemas.SchemaFragment.func` to every matched target. Backend
    fragments carry backend-specific funcs, so core never imports a backend. A fragment
    succeeds when its func returns True on at least one target: each func only tunes its own
    backend's representation and no-ops (returns False) on the other backend's prims, so a
    mixed-backend target set does not fail the write.

    An empty fragment list is an authoring no-op and returns True. When no target matches, a
    warning is emitted and False is returned without authoring anything. Matched prims inside
    instances cannot be authored on and are skipped with a warning.

    Args:
        prim_path_expr: The prim path expression matched against the stage.
        fragments: An iterable of :class:`~isaaclab.sim.schemas.SpatialTendonFragment` instances.
        stage: The stage where to find the prims. Defaults to None, in which case the current
            stage is used.

    Returns:
        True if every fragment tuned at least one target and no instanced prim was skipped.
    """
    return _apply_tendon_fragments(prim_path_expr, fragments, _SPATIAL_TENDON_SCHEMAS, (), "spatial", stage)


"""
Collision mesh properties.
"""


"""
Deformable body properties.
"""


@wp.kernel
def _fix_tet_winding_kernel(
    points: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(ndim=2, dtype=wp.int32),
):
    """Flip any tet with negative signed volume by swapping its last two vertex indices.

    ``UsdGeom.TetMesh`` and :meth:`UsdGeom.TetMesh.ComputeSurfaceFaces` require a
    right-handed tet winding (positive signed volume). Swapping indices 2 and 3
    reverses the orientation without changing which four vertices form the tet.
    """
    i = wp.tid()
    v0 = tet_indices[i, 0]
    v1 = tet_indices[i, 1]
    v2 = tet_indices[i, 2]
    v3 = tet_indices[i, 3]
    p0 = points[v0]
    e1 = points[v1] - p0
    e2 = points[v2] - p0
    e3 = points[v3] - p0
    signed_volume = wp.dot(e1, wp.cross(e2, e3))
    if signed_volume < 0.0:
        tet_indices[i, 2] = v3
        tet_indices[i, 3] = v2


def _tetrahedralize_surface(
    vertices: np.ndarray, faces: np.ndarray, edge_length_fac: float, prim_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedralize a triangle surface mesh with pytetwild, returning right-handed tets.

    Args:
        vertices: Surface vertex positions [m], shape [N, 3].
        faces: Flattened triangle vertex indices, shape [3 * F].
        edge_length_fac: Relative target edge length for the tetrahedralization.
        prim_path: The deformable prim path, used in the error message when the dependency is missing.

    Returns:
        The tetrahedral mesh points [m], shape [M, 3], and tet vertex indices, shape [T, 4].

    Raises:
        ModuleNotFoundError: If the optional tetrahedralization dependencies are not installed.
    """
    try:
        from pytetwild import tetrahedralize
    except ModuleNotFoundError as exc:
        if exc.name not in {"pytetwild", "pyvista", "vtk", "vtkmodules"}:
            raise
        raise ModuleNotFoundError(
            "Automatic tetrahedralization of volume deformables requires the optional "
            "tetrahedralization dependencies. Install them with "
            "uv sync --inexact --extra tetrahedralization from a source checkout, or "
            'uv pip install "isaaclab[tetrahedralization]" from a wheel. Alternatively, provide '
            f"a pre-tetrahedralized UsdGeom.TetMesh under the deformable prim {prim_path}."
        ) from exc

    tet_mesh_points, tet_mesh_indices = tetrahedralize(
        vertices, faces.reshape(-1, 3), edge_length_fac=edge_length_fac, simplify=False, epsilon=1e-2, coarsen=True
    )
    # pytetwild's default ordering does not guarantee positive signed volume, which
    # ``UsdGeom.TetMesh`` and ``ComputeSurfaceFaces`` require. Flip any inverted tets.
    tet_points_wp = wp.array(tet_mesh_points.astype(np.float32), dtype=wp.vec3, device="cpu")
    tet_indices_wp = wp.array(np.asarray(tet_mesh_indices, dtype=np.int32).reshape(-1, 4), dtype=wp.int32, device="cpu")
    wp.launch(
        _fix_tet_winding_kernel, dim=tet_indices_wp.shape[0], inputs=[tet_points_wp, tet_indices_wp], device="cpu"
    )
    return tet_mesh_points, tet_indices_wp.numpy()


def define_deformable_curve_properties(prim_path: str, stage: Usd.Stage | None = None) -> None:
    """Apply the deformable curve simulation schema.

    Args:
        prim_path: The path of the ``UsdGeom.BasisCurves`` prim.
        stage: The stage where the prim exists. Defaults to the current stage.

    Raises:
        ValueError: If the prim path is invalid or is not a ``UsdGeom.BasisCurves`` prim.
        RuntimeError: If the schema cannot be applied.
    """
    if stage is None:
        stage = get_current_stage()

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    if not prim.IsA(UsdGeom.BasisCurves):
        raise ValueError(f"Prim path '{prim_path}' is not a UsdGeom.BasisCurves prim.")

    schema_name = "PhysicsCurvesDeformableSimAPI"
    if schema_name in prim.GetPrimTypeInfo().GetAppliedAPISchemas():
        return
    if not prim.AddAppliedSchema(schema_name):
        raise RuntimeError(f"Failed to set deformable curve API on prim '{prim_path}'.")


def _setup_deformable_meshes(
    root_prim: Usd.Prim,
    deformable_type: str,
    sim_mesh_prim_path: str,
    stage: Usd.Stage,
    tetrahedralization_edge_length_fac: float = 0.1,
) -> tuple[Usd.Prim, Usd.Prim]:
    """Author the backend-neutral simulation and visual meshes for a deformable body.

    This resolves the visual surface mesh under the deformable root prim, creates the simulation
    mesh (a copy of the visual mesh for surface deformables, or a tetrahedralized volume for volume
    deformables), applies the collision API, and hides the simulation mesh from rendering. The
    backend-specific simulation APIs and rest-shape attributes are applied by the caller.

    Args:
        root_prim: The deformable root prim under which to find or author the meshes.
        deformable_type: The type of the deformable body ("surface" or "volume").
        sim_mesh_prim_path: The prim path at which to create the simulation mesh. Ignored when a
            pre-tetrahedralized mesh is found for volume deformables.
        stage: The stage on which the prims live.
        tetrahedralization_edge_length_fac: Relative target edge length for automatic
            tetrahedralization. Defaults to ``0.1``.

    Returns:
        A tuple of the simulation mesh prim and the visual mesh prim.

    Raises:
        ValueError: When the deformable type is unsupported, no mesh or multiple meshes are found,
            or a resolved mesh prim is invalid.
        RuntimeError: When applying the collision API fails.

    """
    if deformable_type not in ("surface", "volume"):
        raise ValueError(f"Unsupported deformable type: '{deformable_type}'. Expected surface or volume.")

    prim_path = str(root_prim.GetPrimPath())

    # volume deformables may ship a pre-tetrahedralized TetMesh to use as the simulation mesh
    sim_mesh_prim = None
    if deformable_type == "volume":
        matching_prims = get_all_matching_child_prims(prim_path, lambda p: p.GetTypeName() == "TetMesh", stage=stage)
        if len(matching_prims) > 1:
            mesh_paths = [p.GetPrimPath() for p in matching_prims]
            raise ValueError(
                f"Found multiple tetrahedral meshes in '{prim_path}': {mesh_paths}."
                " Deformable body schema can only be applied to one mesh for now."
            )
        if matching_prims:
            sim_mesh_prim = matching_prims[0]
            if not sim_mesh_prim.IsValid():
                raise ValueError(f"Mesh prim path '{sim_mesh_prim.GetPrimPath()}' is not valid.")

    matching_prims = get_all_matching_child_prims(prim_path, lambda p: p.GetTypeName() == "Mesh", stage=stage)
    if len(matching_prims) == 0:
        # with a TetMesh but no Mesh, the TetMesh surface becomes the visual mesh
        if sim_mesh_prim is not None:
            tet_mesh_prim = UsdGeom.TetMesh(sim_mesh_prim)
            surface_indices = UsdGeom.TetMesh.ComputeSurfaceFaces(tet_mesh_prim, Usd.TimeCode.Default())
            if surface_indices is None or len(surface_indices) == 0:
                raise ValueError(
                    f"Deformable body at '{prim_path}' has no surface indices on its TetMesh prim; "
                    "cannot sync to visual mesh."
                )
            vis_mesh_prim = create_prim(
                prim_path + "/vis_mesh",
                prim_type="Mesh",
                attributes={
                    "points": tet_mesh_prim.GetPointsAttr().Get(),
                    "faceVertexIndices": np.asarray(surface_indices).flatten(),
                    "faceVertexCounts": [3] * len(surface_indices),
                },
                stage=stage,
            )
            matching_prims = [vis_mesh_prim]
        else:
            raise ValueError(f"Could not find any visual mesh in '{prim_path}'. Please check asset.")
    if len(matching_prims) > 1:
        mesh_paths = [p.GetPrimPath() for p in matching_prims]
        raise ValueError(
            f"Found multiple visual meshes in '{prim_path}': {mesh_paths}."
            " Deformable body schema can only be applied to one mesh for now."
        )
    vis_mesh_prim = matching_prims[0]
    if not vis_mesh_prim.IsValid():
        raise ValueError(f"Mesh prim path '{vis_mesh_prim.GetPrimPath()}' is not valid.")

    # extract visual surface mesh vertices and faces
    vertices = np.array(vis_mesh_prim.GetAttribute("points").Get())
    faces = np.array(vis_mesh_prim.GetAttribute("faceVertexIndices").Get()).flatten()
    face_counts = np.array(vis_mesh_prim.GetAttribute("faceVertexCounts").Get())
    if deformable_type == "surface":
        # the simulation mesh is a copy of the visual mesh
        sim_mesh_prim = create_prim(
            sim_mesh_prim_path,
            prim_type="Mesh",
            attributes={
                "points": vertices,
                "faceVertexIndices": faces,
                "faceVertexCounts": face_counts,
            },
            stage=stage,
        )
    else:
        if sim_mesh_prim is None:
            tet_mesh_points, tet_mesh_indices = _tetrahedralize_surface(
                vertices, faces, tetrahedralization_edge_length_fac, prim_path
            )
            sim_mesh_prim = create_prim(
                sim_mesh_prim_path,
                prim_type="TetMesh",
                attributes={
                    "points": tet_mesh_points,
                    "tetVertexIndices": tet_mesh_indices,
                },
                stage=stage,
            )

        # set surface faces required by the deformable simulation APIs
        surface_face_indices = UsdGeom.TetMesh.ComputeSurfaceFaces(
            UsdGeom.TetMesh(sim_mesh_prim), Usd.TimeCode.Default()
        )
        UsdGeom.TetMesh(sim_mesh_prim).GetSurfaceFaceVertexIndicesAttr().Set(surface_face_indices)

    if not sim_mesh_prim.ApplyAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"Failed to set {deformable_type} deformable collision API on prim '{sim_mesh_prim_path}'.")
    # the simulation mesh is not rendered
    UsdGeom.Imageable(sim_mesh_prim).GetPurposeAttr().Set(UsdGeom.Tokens.guide)

    return sim_mesh_prim, vis_mesh_prim


def _setup_omniphysics_deformable_body(
    prim: Usd.Prim, deformable_type: str, sim_mesh_prim: Usd.Prim, vis_mesh_prim: Usd.Prim
) -> None:
    """Apply the OmniPhysics deformable anchor schemas, rest state, and bind pose to a prepared body.

    The OmniPhysics deformable schemas are shared by the PhysX-based backends (Kit PhysX and
    OvPhysX), so their :meth:`~isaaclab.physics.PhysicsManager.setup_deformable_body` hooks delegate
    here after :func:`_setup_deformable_meshes` has created the simulation and visual meshes.

    Args:
        prim: The deformable-body prim to anchor.
        deformable_type: The deformable type, ``"volume"`` or ``"surface"``.
        sim_mesh_prim: The prepared simulation-mesh prim.
        vis_mesh_prim: The visual-mesh prim.

    Raises:
        RuntimeError: When an anchor schema cannot be applied.
    """
    sim_mesh_path = sim_mesh_prim.GetPath().pathString
    if deformable_type == "surface":
        if not sim_mesh_prim.ApplyAPI("OmniPhysicsSurfaceDeformableSimAPI"):
            raise RuntimeError(f"Failed to set surface deformable sim API on prim '{sim_mesh_path}'.")
        sim_mesh_prim.GetAttribute("omniphysics:restShapePoints").Set(sim_mesh_prim.GetAttribute("points").Get())
        # flatten through numpy so USD coerces the flat index run into the Vec3i array the
        # schema declares; a ``Vt.IntArray`` read straight back is rejected as a type mismatch
        sim_mesh_prim.GetAttribute("omniphysics:restTriVtxIndices").Set(
            np.asarray(sim_mesh_prim.GetAttribute("faceVertexIndices").Get()).flatten()
        )
    else:
        if not sim_mesh_prim.ApplyAPI("OmniPhysicsVolumeDeformableSimAPI"):
            raise RuntimeError(f"Failed to set volume deformable sim API on prim '{sim_mesh_path}'.")
        sim_mesh_prim.GetAttribute("omniphysics:restShapePoints").Set(sim_mesh_prim.GetAttribute("points").Get())
        sim_mesh_prim.GetAttribute("omniphysics:restTetVtxIndices").Set(
            sim_mesh_prim.GetAttribute("tetVertexIndices").Get()
        )
    # bind visual to sim mesh through the default bind pose
    purposes = ["bindPose"]
    vis_mesh_prim.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
    vis_mesh_prim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(
        purposes
    )
    points = UsdGeom.PointBased(vis_mesh_prim).GetPointsAttr().Get()
    vis_mesh_prim.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(
        points
    )
    sim_mesh_prim.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
    sim_mesh_prim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(
        purposes
    )
    if not prim.ApplyAPI("OmniPhysicsDeformableBodyAPI"):
        raise RuntimeError(f"Failed to set deformable body API on prim '{prim.GetPath().pathString}'.")
