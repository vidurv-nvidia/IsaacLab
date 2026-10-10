# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import ClassVar, Literal

from ...utils import configclass


def __getattr__(name):
    # ``NewtonMaterialPropertiesCfg`` moved to ``isaaclab_newton.sim.schemas.schemas_cfg``; resolved
    # lazily so this module does not import ``isaaclab_newton`` at load time.
    if name == "NewtonMaterialPropertiesCfg":
        try:
            from isaaclab_newton.sim.schemas import schemas_cfg as _newton_cfg
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.schemas.schemas_cfg.{name}' has moved to"
                " 'isaaclab_newton.sim.schemas.schemas_cfg'. Install the isaaclab_newton"
                " extension or update your import. This forwarding shim is scheduled for"
                " removal in 4.0."
            ) from e
        return getattr(_newton_cfg, name)
    raise AttributeError(f"module 'isaaclab.sim.schemas.schemas_cfg' has no attribute {name!r}")


def deprecate_field_alias(cfg, alias: str, canonical: str) -> None:
    """Forward a deprecated cfg field to its canonical replacement.

    If ``alias`` is set on the cfg instance, emit a ``DeprecationWarning`` and copy the
    value to ``canonical`` (when ``canonical`` is unset). The alias is then nulled so
    downstream metadata-driven writers see only the canonical name.
    """
    value = getattr(cfg, alias, None)
    if value is None:
        return
    warnings.warn(
        f"'{alias}' is deprecated; use '{canonical}' instead. The alias is scheduled for removal in 4.0.",
        DeprecationWarning,
        stacklevel=3,
    )
    if getattr(cfg, canonical, None) is None:
        setattr(cfg, canonical, value)
    setattr(cfg, alias, None)


@configclass
class SchemaFragment:
    """Base for a single-namespace USD-schema config fragment.

    Each subclass mirrors exactly one USD applied schema. The fragment carries class-level
    metadata describing which USD namespace its fields write to (:attr:`_usd_namespace`) and
    which applied schema, if any, it owns (:attr:`_usd_applied_schema`). The :attr:`func`
    field names the callable that applies the fragment to a prim; the default generic applier
    (:func:`~isaaclab.sim.schemas.apply_namespaced`) reads the metadata and writes each
    non-``None`` field as ``<namespace>:<camelCase(field)>``. Irregular APIs override
    :attr:`func` with a custom applier.

    .. note::
        A fragment present in a spawner slot means its schema is applied. ``None`` fields are
        left unchanged on the prim (partial update).

    .. important::
        For fragments using :func:`~isaaclab.sim.schemas.apply_namespaced`, every dataclass field
        other than :attr:`func` is authored as ``<_usd_namespace>:<camelCase(field)>``. Irregular
        schemas may use a custom applier for value conversion. The PhysX tendon fragments are a
        narrow exception: their custom appliers consume ``instance_names`` to address an existing
        multiple-apply schema instance and never author it. Do not add bookkeeping fields or treat
        multiple-apply behavior as a generic core-fragment convention.
    """

    # -- Class metadata (not dataclass fields) --
    _usd_namespace: ClassVar[str | None] = None
    _usd_applied_schema: ClassVar[str | None] = None

    func: Callable | str = "isaaclab.sim.schemas:apply_namespaced"
    """Callable (or its ``module:attr`` import string) that applies this fragment to a prim.

    Resolved via :func:`~isaaclab.utils.string.string_to_callable` when a string. The callable
    signature is ``func(cfg, prim_path, stage)``.
    """


@configclass
class RigidBodyFragment(SchemaFragment):
    """Marker base for rigid-body fragments; types the ``rigid_props`` slot."""


@configclass
class UsdPhysicsRigidBodyCfg(RigidBodyFragment):
    """``physics:*`` rigid-body attributes from `UsdPhysics.RigidBodyAPI`_.

    The ``UsdPhysics.RigidBodyAPI`` schema is applied as the implicit anchor by the rigid-body
    family writer, so this fragment owns no applied schema of its own.

    .. _UsdPhysics.RigidBodyAPI: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = None  # RigidBodyAPI applied by the family anchor

    rigid_body_enabled: bool | None = None
    """Whether to enable or disable the rigid body."""

    kinematic_enabled: bool | None = None
    """Determines whether the body is kinematic or not.

    A kinematic body is moved through animated or user-defined poses; the simulation still
    derives velocities for it based on the external motion.
    """


@configclass
class CollisionFragment(SchemaFragment):
    """Marker base for collision fragments; types the ``collision_props`` slot."""


@configclass
class ArticulationRootFragment(SchemaFragment):
    """Marker base for articulation-root fragments; types the ``articulation_props`` slot.

    Articulation-root fragments author backend-specific articulation properties (solver
    iterations, sleep / stabilization thresholds, self-collision toggles). The defining
    ``UsdPhysics.ArticulationRootAPI`` anchor is applied by the articulation-root family
    writer (:func:`~isaaclab.sim.schemas.apply_articulation_root_properties`) only when the
    ``articulation_props`` slot carries fragments (presence-gated).
    """


@configclass
class JointDriveFragment(SchemaFragment):
    """Marker base for joint-drive fragments; types the ``joint_drive_props`` slot."""


@configclass
class MeshCollisionFragment(SchemaFragment):
    """Marker base for mesh-collision fragments; types the ``mesh_collision_props`` slot.

    A mesh-collision concept is split across one *core* fragment carrying the standard
    ``physics:approximation`` token (:class:`UsdPhysicsMeshCollisionCfg`) and one cooking
    fragment per backend cooking schema (PhysX convex hull / decomposition / triangle mesh /
    SDF, Newton mesh / SDF). A PhysX cooking fragment implies the approximation token written to
    ``physics:approximation`` through its default ``mesh_approximation_name``; the Newton cooking
    fragments carry no token and leave it unchanged -- see
    :func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`.
    """

    # Mesh-collision fragments author the shared ``physics:approximation`` token in addition to their
    # own namespaced cooking attrs, so they dispatch through :func:`~isaaclab.sim.schemas.apply_mesh_collision`
    # (not the generic :func:`~isaaclab.sim.schemas.apply_namespaced`). See that func for the token coupling.
    func: Callable | str = "isaaclab.sim.schemas:apply_mesh_collision"


@configclass
class FixedTendonFragment(SchemaFragment):
    """Marker base for fixed-tendon fragments; types the ``fixed_tendons_props`` slot.

    Fixed tendons are a *tune-not-apply* family: the applied ``PhysxTendonAxisRootAPI`` and
    ``PhysxTendonAxisAPI`` instances already exist on joint prims (authored in the source asset),
    so the family writer (:func:`~isaaclab.sim.schemas.apply_fixed_tendon_properties`) does not
    apply an anchor schema; it only tunes existing instances via each fragment's
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func`.
    """


@configclass
class SpatialTendonFragment(SchemaFragment):
    """Marker base for spatial-tendon fragments; types the ``spatial_tendons_props`` slot.

    Spatial tendons are a *tune-not-apply* family: the applied
    ``PhysxTendonAttachmentRootAPI`` instances already exist on the prim (authored in the source
    asset), so the family writer (:func:`~isaaclab.sim.schemas.apply_spatial_tendon_properties`)
    does not apply an anchor schema; it only tunes existing root instances via each fragment's
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func`.
    """


@configclass
class DeformableBodyFragment(SchemaFragment):
    """Marker base for deformable-body fragments; types the ``volume_deformable_props`` and
    ``surface_deformable_props`` slots.

    The deformable anchor schemas (sim and body APIs) are applied by the family writers through
    the active physics backend, so fragments never own the anchor. A fragment meaningful for only
    one deformable type narrows :attr:`_deformable_types`; the family writers warn (but still
    author) when a fragment is passed to the other type's writer.
    """

    _deformable_types: ClassVar[tuple[str, ...]] = ("volume", "surface")


@configclass
class OmniPhysicsDeformableBodyCfg(DeformableBodyFragment):
    """``omniphysics:*`` deformable-body attributes from ``OmniPhysicsDeformableBodyAPI``.

    The ``OmniPhysicsDeformableBodyAPI`` anchor is applied by the family writer through the
    active physics backend, so this fragment owns no applied schema of its own.
    """

    _usd_namespace: ClassVar[str | None] = "omniphysics"
    _usd_applied_schema: ClassVar[str | None] = None  # anchor applied by the backend manager

    deformable_body_enabled: bool | None = None
    """Whether the deformable body participates in the simulation."""

    kinematic_enabled: bool | None = None
    """Whether the body is kinematic (driven by animated or user-defined poses)."""

    mass: float | None = None
    """The body mass [kg]. Overrides material-density-derived mass; ``UsdPhysics.MassAPI`` is
    ignored for deformable bodies."""


@configclass
class UsdPhysicsCollisionCfg(CollisionFragment):
    """``physics:*`` collision attributes from `UsdPhysics.CollisionAPI`_.

    The ``UsdPhysics.CollisionAPI`` schema is applied as the implicit anchor by the collision
    family writer (:func:`~isaaclab.sim.schemas.apply_collision_properties`), so this fragment
    owns no applied schema of its own.

    .. _UsdPhysics.CollisionAPI: https://openusd.org/dev/api/class_usd_physics_collision_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = None  # CollisionAPI applied by the family anchor

    collision_enabled: bool | None = None
    """Whether to enable or disable collisions.

    Writes ``physics:collisionEnabled`` via :class:`UsdPhysics.CollisionAPI`.
    """


@configclass
class UsdPhysicsDriveCfg(JointDriveFragment):
    """``drive:<linear|angular>:physics:*`` joint-drive attributes from `UsdPhysics.DriveAPI`_.

    The drive attributes live under a multi-instance ``UsdPhysics.DriveAPI`` (instance
    ``"angular"`` for revolute joints, ``"linear"`` for prismatic joints), so this fragment
    cannot use the generic :func:`~isaaclab.sim.schemas.apply_namespaced` writer. It overrides
    :attr:`func` with :func:`~isaaclab.sim.schemas.apply_drive`, which selects the instance,
    applies ``UsdPhysics.DriveAPI`` (presence-gated, the conditional anchor for the joint-drive
    family), performs the radian-to-degree conversion for angular drives, and writes the typed
    ``drive:<inst>:physics:{type,maxForce,stiffness,damping}`` attributes.

    .. note::
        Unlike most fragments, this one is not a metadata-driven write. ``DriveAPI`` is applied
        only when this fragment is present in the slot.

    .. _UsdPhysics.DriveAPI: https://openusd.org/dev/api/class_usd_physics_drive_a_p_i.html
    """

    # No metadata-driven namespace: the typed multi-instance ``UsdPhysics.DriveAPI`` is written
    # directly by ``apply_drive``. ``DriveAPI`` is presence-gated, not an implicit anchor.
    _usd_namespace: ClassVar[str | None] = None
    _usd_applied_schema: ClassVar[str | None] = None

    func: Callable | str = "isaaclab.sim.schemas:apply_drive"

    def __post_init__(self):
        # Deprecation alias: ``max_effort`` -> ``max_force`` (the USD attr is ``maxForce``).
        deprecate_field_alias(self, "max_effort", "max_force")

    drive_type: Literal["force", "acceleration"] | None = None
    """Joint drive type to apply.

    If the drive type is ``"force"``, then the joint is driven by a force. If the drive type is
    ``"acceleration"``, then the joint is driven by an acceleration (usually used for kinematic
    joints). Written to ``drive:<inst>:physics:type`` (the USD attr is ``type``, a permanent
    inline carve-out from the snake-to-camel convention).
    """

    max_force: float | None = None
    """Maximum force/torque that can be applied to the joint [N for linear joints, N·m for angular joints].

    Written to ``drive:<inst>:physics:maxForce`` via :class:`UsdPhysics.DriveAPI`.
    """

    max_effort: float | None = None
    """Deprecated alias for :attr:`max_force`.

    .. deprecated:: 3.1
        Use :attr:`max_force` instead. The cfg field is renamed so its snake_case name maps
        identity-style to the USD camelCase attribute (``maxForce`` on ``UsdPhysics.DriveAPI``).
        The alias is forwarded to :attr:`max_force` in :meth:`__post_init__` and will be removed
        in 3.2.
    """

    stiffness: float | None = None
    """Stiffness of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s² (N/m).
    * For angular joints, the unit is kg-m²/s²/rad (N·m/rad).

    Angular drives are converted from radians to degrees (``N·m/rad`` -> ``N·m/deg``) before
    being written to ``drive:angular:physics:stiffness``.
    """

    damping: float | None = None
    """Damping of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s (N·s/m).
    * For angular joints, the unit is kg-m²/s/rad (N·m·s/rad).

    Angular drives are converted from radians to degrees (``N·m·s/rad`` -> ``N·m·s/deg``) before
    being written to ``drive:angular:physics:damping``.
    """


@configclass
class UsdPhysicsMeshCollisionCfg(MeshCollisionFragment):
    """``physics:approximation`` mesh-collision token from `UsdPhysics.MeshCollisionAPI`_.

    Carries the standard mesh-collision approximation token (:attr:`mesh_approximation_name`
    written to ``physics:approximation``). The ``UsdPhysics.MeshCollisionAPI`` schema is applied
    as the implicit anchor by the mesh-collision family writer
    (:func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`), so this fragment owns no
    applied schema of its own.

    .. note::
        The ``physics:approximation`` attribute is a ``TfToken`` validated against
        :const:`~isaaclab.sim.schemas.MESH_APPROXIMATION_TOKENS`; the family writer (not the generic
        :func:`~isaaclab.sim.schemas.apply_namespaced` applier) handles the token write, so this
        fragment overrides nothing but the namespace metadata. When a PhysX/Newton cooking fragment
        is present alongside this one, its default :attr:`mesh_approximation_name` sets the token.

    .. _UsdPhysics.MeshCollisionAPI: https://openusd.org/release/api/class_usd_physics_mesh_collision_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = None  # MeshCollisionAPI applied by the family anchor

    mesh_approximation_name: str = "none"
    """Name of mesh collision approximation method. Default: "none".

    Writes the ``physics:approximation`` token via :class:`UsdPhysics.MeshCollisionAPI`.
    Refer to :const:`~isaaclab.sim.schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """


@configclass
class MassFragment(SchemaFragment):
    """Marker base for mass fragments; types the ``mass_props`` slot."""


@configclass
class MassCfg(MassFragment):
    """``physics:*`` mass attributes from `UsdPhysics.MassAPI`_.

    The ``UsdPhysics.MassAPI`` schema is applied as the implicit anchor by the mass family writer
    (:func:`~isaaclab.sim.schemas.apply_mass_properties`), so this fragment owns no applied schema
    of its own.

    .. note::
        A fragment present in a spawner slot means its schema is applied. ``None`` fields are left
        unchanged on the prim (partial update).

    .. _UsdPhysics.MassAPI: https://openusd.org/dev/api/class_usd_physics_mass_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = None  # MassAPI applied by the family anchor

    mass: float | None = None
    """The mass of the rigid body [kg].

    Writes ``physics:mass`` via :class:`UsdPhysics.MassAPI`.

    Note:
        If ``density`` is non-zero, it takes precedence and is used to compute the mass instead.
    """

    density: float | None = None
    """The density of the rigid body [kg/m^3].

    Writes ``physics:density`` via :class:`UsdPhysics.MassAPI`. The density indirectly defines the
    mass of the rigid body. It is generally computed using the collision approximation of the body.
    """
