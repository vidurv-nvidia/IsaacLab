# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Literal

from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootFragment,
    CollisionFragment,
    DeformableBodyFragment,
    FixedTendonFragment,
    JointDriveFragment,
    MeshCollisionFragment,
    RigidBodyFragment,
    SpatialTendonFragment,
    deprecate_field_alias,
)
from isaaclab.utils import configclass


@configclass
class PhysxDeformableBodyCfg(DeformableBodyFragment):
    """``physxDeformableBody:*`` deformable-body attributes from `PhysxBaseDeformableBodyAPI`_.

    A single-namespace fragment (see :class:`~isaaclab.sim.schemas.SchemaFragment`) covering the
    solver, damping, and self-collision attributes shared by volume and surface deformables.

    .. _PhysxBaseDeformableBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/physx_deformable_schema.html#physxbasedeformablebodyapi
    """

    _usd_namespace: ClassVar[str | None] = "physxDeformableBody"
    _usd_applied_schema: ClassVar[str | None] = "PhysxBaseDeformableBodyAPI"

    solver_position_iteration_count: int | None = None
    """Number of solver position iterations."""

    linear_damping: float | None = None
    """Linear velocity damping [1/s]."""

    max_linear_velocity: float | None = None
    """Maximum linear velocity [m/s]."""

    settling_damping: float | None = None
    """Damping applied while the body settles [1/s]."""

    settling_threshold: float | None = None
    """Velocity threshold below which settling damping engages [m/s]."""

    sleep_threshold: float | None = None
    """Velocity threshold below which the body may sleep [m/s]."""

    max_depenetration_velocity: float | None = None
    """Maximum velocity used to resolve penetrations [m/s]."""

    self_collision: bool | None = None
    """Whether the body collides with itself."""

    self_collision_filter_distance: float | None = None
    """Distance under which self-collision contacts are filtered [m]."""

    enable_speculative_c_c_d: bool | None = None
    """Whether speculative continuous collision detection is enabled."""

    disable_gravity: bool | None = None
    """Whether gravity is disabled for this body."""


@configclass
class PhysxSurfaceDeformableBodyCfg(DeformableBodyFragment):
    """``physxDeformableBody:*`` surface-only attributes from `PhysxSurfaceDeformableBodyAPI`_.

    A single-namespace fragment (see :class:`~isaaclab.sim.schemas.SchemaFragment`) narrowed to
    surface deformables via :attr:`~isaaclab.sim.schemas.DeformableBodyFragment._deformable_types`.

    .. _PhysxSurfaceDeformableBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/physx_deformable_schema.html#physxsurfacedeformablebodyapi
    """

    _usd_namespace: ClassVar[str | None] = "physxDeformableBody"
    _usd_applied_schema: ClassVar[str | None] = "PhysxSurfaceDeformableBodyAPI"
    _deformable_types: ClassVar[tuple[str, ...]] = ("surface",)

    collision_pair_update_frequency: int | None = None
    """How often collision pairs are refreshed, in solver steps."""

    collision_iteration_multiplier: float | None = None
    """Multiplier on collision solver iterations."""


@configclass
class PhysxRigidBodyCfg(RigidBodyFragment):
    """``physxRigidBody:*`` rigid-body attributes from `PhysxRigidBodyAPI`_.

    A single-namespace fragment (see :class:`~isaaclab.sim.schemas.SchemaFragment`) for the
    PhysX rigid-body add-on schema. Applied alongside :class:`~isaaclab.sim.schemas.UsdPhysicsRigidBodyCfg`
    via :func:`~isaaclab.sim.schemas.apply_rigid_body_properties`.

    .. _PhysxRigidBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_rigid_body_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physxRigidBody"
    _usd_applied_schema: ClassVar[str | None] = "PhysxRigidBodyAPI"

    linear_damping: float | None = None
    """Linear damping coefficient for the body [1/s]."""

    angular_damping: float | None = None
    """Angular damping coefficient for the body [1/s]."""

    max_linear_velocity: float | None = None
    """Maximum linear velocity for the body [m/s]."""

    max_angular_velocity: float | None = None
    """Maximum angular velocity for the body [deg/s]."""

    max_depenetration_velocity: float | None = None
    """Maximum depenetration velocity permitted to be introduced by the solver [m/s]."""

    max_contact_impulse: float | None = None
    """The limit on the impulse that may be applied at a contact [N·s]."""

    enable_gyroscopic_forces: bool | None = None
    """Enables computation of gyroscopic forces on the rigid body."""

    retain_accelerations: bool | None = None
    """Carries over forces/accelerations over sub-steps."""

    solver_position_iteration_count: int | None = None
    """Solver position iteration counts for the body."""

    solver_velocity_iteration_count: int | None = None
    """Solver velocity iteration counts for the body."""

    sleep_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an actor may go to sleep [m²/s²]."""

    stabilization_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an actor may participate in stabilization [m²/s²]."""

    disable_gravity: bool | None = None
    """Disable gravity for the body.

    PhysX honors this per-body via ``physxRigidBody:disableGravity``: setting True excludes the
    body from world gravity integration.
    """


@configclass
class PhysxJointCfg(JointDriveFragment):
    """``physxJoint:*`` joint attributes from `PhysxJointAPI`_.

    A single-namespace fragment (see :class:`~isaaclab.sim.schemas.SchemaFragment`) for the
    PhysX joint add-on schema. Applied alongside :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg`
    via :func:`~isaaclab.sim.schemas.apply_joint_drive_properties`. Written with the dedicated
    :func:`~isaaclab_physx.sim.schemas.apply_physx_joint` writer, which converts
    :attr:`max_joint_velocity` from rad/s to deg/s for angular (revolute) joints.

    .. _PhysxJointAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_joint_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physxJoint"
    _usd_applied_schema: ClassVar[str | None] = "PhysxJointAPI"
    # Override the generic applier: ``max_joint_velocity`` needs joint-type-aware rad->deg
    # conversion for angular joints, which ``apply_namespaced`` cannot do.
    func: Callable | str = "isaaclab_physx.sim.schemas:apply_physx_joint"

    def __post_init__(self):
        # Deprecation alias: ``max_velocity`` -> ``max_joint_velocity`` (the USD attr is
        # ``maxJointVelocity``).
        deprecate_field_alias(self, "max_velocity", "max_joint_velocity")

    max_joint_velocity: float | None = None
    """Maximum velocity of the joint [m/s for linear joints, rad/s for angular joints].

    Notes:
        Today this writes ``physxJoint:maxJointVelocity`` (a PhysX add-on schema attribute).
        Newton's USD importer consumes the same attribute via its PhysX-bridge resolver and
        populates ``Model.joint_velocity_limit``; the PhysX engine consumes it natively.

    .. note::
        Authored in rad/s; :func:`~isaaclab_physx.sim.schemas.apply_physx_joint` converts it to
        deg/s for angular (revolute) joints (PhysX's angular convention) and writes linear
        (prismatic) joints unchanged.
    """

    max_velocity: float | None = None
    """Deprecated alias for :attr:`max_joint_velocity`.

    .. deprecated:: 3.1
        Use :attr:`max_joint_velocity` instead. The cfg field is renamed so its snake_case name
        maps identity-style to the USD camelCase attribute (``physxJoint:maxJointVelocity``). The
        alias is forwarded to :attr:`max_joint_velocity` in :meth:`__post_init__` and will be
        removed in 3.2.
    """


@configclass
class PhysxCollisionCfg(CollisionFragment):
    """``physxCollision:*`` collision attributes from `PhysxCollisionAPI`_.

    A single-namespace fragment (see :class:`~isaaclab.sim.schemas.SchemaFragment`) for the
    PhysX collision add-on schema. Applied alongside
    :class:`~isaaclab.sim.schemas.UsdPhysicsCollisionCfg` via
    :func:`~isaaclab.sim.schemas.apply_collision_properties`.

    The :attr:`contact_offset` / :attr:`rest_offset` knobs live here as plain
    ``physxCollision:*`` fields. Newton's USD importer consumes the same attributes via its
    PhysX-bridge resolver, so they are not duplicated on the Newton collision fragment.

    .. _PhysxCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_collision_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physxCollision"
    _usd_applied_schema: ClassVar[str | None] = "PhysxCollisionAPI"

    contact_offset: float | None = None
    """Contact offset for the collision shape [m].

    The collision detector generates contact points as soon as two shapes get closer than the sum
    of their contact offsets. This quantity should be non-negative, which means contact generation
    can potentially start before the shapes actually penetrate.

    Writes ``physxCollision:contactOffset``. Newton's USD importer consumes the same attribute via
    its PhysX-bridge resolver.
    """

    rest_offset: float | None = None
    """Rest offset for the collision shape [m].

    The rest offset quantifies how close a shape gets to others at rest. At rest, the distance
    between two vertically stacked objects is the sum of their rest offsets. If a pair of shapes
    have a positive rest offset, the shapes will be separated at rest by an air gap.

    Writes ``physxCollision:restOffset``. Newton's USD importer consumes the same attribute via its
    PhysX-bridge resolver.
    """

    torsional_patch_radius: float | None = None
    """Radius of the contact patch for applying torsional friction [m].

    It is used to approximate rotational friction introduced by the compression of contacting
    surfaces. If the radius is zero, no torsional friction is applied.
    """

    min_torsional_patch_radius: float | None = None
    """Minimum radius of the contact patch for applying torsional friction [m]."""


@configclass
class PhysxArticulationCfg(ArticulationRootFragment):
    """``physxArticulation:*`` articulation-root attributes from `PhysxArticulationAPI`_.

    A single-namespace fragment (see :class:`~isaaclab.sim.schemas.SchemaFragment`) for the PhysX
    articulation add-on schema. Applied alongside other articulation-root fragments via
    :func:`~isaaclab.sim.schemas.apply_articulation_root_properties`, which applies the
    ``UsdPhysics.ArticulationRootAPI`` anchor (presence-gated). This fragment owns the
    ``PhysxArticulationAPI`` applied schema.

    .. _PhysxArticulationAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_articulation_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physxArticulation"
    _usd_applied_schema: ClassVar[str | None] = "PhysxArticulationAPI"

    articulation_enabled: bool | None = None
    """Whether to enable or disable the articulation.

    PhysX honors this per-articulation at sim time via ``physxArticulation:articulationEnabled``:
    setting False makes PhysX skip the articulation in its solver passes.
    """

    enabled_self_collisions: bool | None = None
    """Whether self-collisions between bodies in the same articulation are enabled.

    Written to ``physxArticulation:enabledSelfCollisions``. The Newton-native counterpart is
    :attr:`~isaaclab_newton.sim.schemas.NewtonArticulationCfg.self_collision_enabled`
    (``newton:selfCollisionEnabled``).
    """

    solver_position_iteration_count: int | None = None
    """Solver position iteration counts for the articulation."""

    solver_velocity_iteration_count: int | None = None
    """Solver velocity iteration counts for the articulation."""

    sleep_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an actor may go to sleep [m²/s²]."""

    stabilization_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an articulation may participate in
    stabilization [m²/s²]."""


# -------------------------------------------------------------------------------------
# Mesh-collision cooking fragments (single-namespace; PhysX cooking add-on schemas).
#
# Each fragment owns one ``physx*Collision:*`` namespace + applied schema; its
# ``mesh_approximation_name`` default encodes the ``physics:approximation`` token its cooking
# schema implies. The token is written by the family writer
# ``isaaclab.sim.schemas.apply_mesh_collision_properties``; tuning attrs go via ``apply_namespaced``.
# -------------------------------------------------------------------------------------


@configclass
class PhysxConvexHullCfg(MeshCollisionFragment):
    """``physxConvexHullCollision:*`` mesh-cooking attributes from `PhysxConvexHullCollisionAPI`_.

    A single-namespace fragment (see :class:`~isaaclab.sim.schemas.SchemaFragment`) for the PhysX
    convex-hull cooking schema. The ``convexHull`` token is written to ``physics:approximation`` by
    :func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`.

    .. _PhysxConvexHullCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_convex_hull_collision_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physxConvexHullCollision"
    _usd_applied_schema: ClassVar[str | None] = "PhysxConvexHullCollisionAPI"

    mesh_approximation_name: str = "convexHull"
    """Name of mesh collision approximation method. Default: "convexHull"."""

    hull_vertex_limit: int | None = None
    """Convex hull vertex limit used for convex hull cooking [dimensionless]. Defaults to 64."""

    min_thickness: float | None = None
    """Convex hull min thickness [m]. Range: [0, inf). Default value is 0.001."""


@configclass
class PhysxConvexDecompositionCfg(MeshCollisionFragment):
    """``physxConvexDecompositionCollision:*`` mesh-cooking attributes from `PhysxConvexDecompositionCollisionAPI`_.

    A single-namespace fragment for the PhysX convex-decomposition cooking schema. The
    ``convexDecomposition`` token is written to ``physics:approximation`` by
    :func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`.

    .. _PhysxConvexDecompositionCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_convex_decomposition_collision_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physxConvexDecompositionCollision"
    _usd_applied_schema: ClassVar[str | None] = "PhysxConvexDecompositionCollisionAPI"

    mesh_approximation_name: str = "convexDecomposition"
    """Name of mesh collision approximation method. Default: "convexDecomposition"."""

    hull_vertex_limit: int | None = None
    """Convex hull vertex limit used for convex hull cooking [dimensionless]. Defaults to 64."""

    max_convex_hulls: int | None = None
    """Maximum of convex hulls created during convex decomposition [dimensionless]. Default value is 32."""

    min_thickness: float | None = None
    """Convex hull min thickness [m]. Range: [0, inf). Default value is 0.001."""

    voxel_resolution: int | None = None
    """Voxel resolution used for convex decomposition [dimensionless]. Defaults to 500,000 voxels."""

    error_percentage: float | None = None
    """Convex decomposition error percentage parameter [%]. Defaults to 10 percent."""

    shrink_wrap: bool | None = None
    """Attempts to adjust the convex hull points so that they are projected onto the surface of the
    original graphics mesh. Defaults to False.
    """


@configclass
class PhysxTriangleMeshCfg(MeshCollisionFragment):
    """``physxTriangleMeshCollision:*`` mesh-cooking attributes from `PhysxTriangleMeshCollisionAPI`_.

    A single-namespace fragment for the PhysX triangle-mesh cooking schema (PhysX-only colliders).

    .. _PhysxTriangleMeshCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_triangle_mesh_collision_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physxTriangleMeshCollision"
    _usd_applied_schema: ClassVar[str | None] = "PhysxTriangleMeshCollisionAPI"

    mesh_approximation_name: str = "none"
    """Name of mesh collision approximation method. Default: "none" (uses triangle mesh)."""

    weld_tolerance: float | None = None
    """Mesh weld tolerance controlling the distance at which vertices are welded [m].

    Default ``-inf`` autocomputes the welding tolerance from the mesh size; ``0`` disables welding.
    Range: [0, inf).
    """


@configclass
class PhysxTriangleMeshSimplificationCfg(MeshCollisionFragment):
    """``physxTriangleMeshSimplificationCollision:*`` attributes from `PhysxTriangleMeshSimplificationCollisionAPI`_.

    A single-namespace fragment for the PhysX triangle-mesh-simplification cooking schema. The
    ``meshSimplification`` token is written to ``physics:approximation`` by
    :func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`.

    .. _PhysxTriangleMeshSimplificationCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_triangle_mesh_simplification_collision_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physxTriangleMeshSimplificationCollision"
    _usd_applied_schema: ClassVar[str | None] = "PhysxTriangleMeshSimplificationCollisionAPI"

    mesh_approximation_name: str = "meshSimplification"
    """Name of mesh collision approximation method. Default: "meshSimplification"."""

    simplification_metric: float | None = None
    """Mesh simplification accuracy [dimensionless]. Defaults to 0.55."""

    weld_tolerance: float | None = None
    """Mesh weld tolerance controlling the distance at which vertices are welded [m].

    Default ``-inf`` autocomputes the welding tolerance from the mesh size; ``0`` disables welding.
    Range: [0, inf).
    """


@configclass
class PhysxSDFMeshCfg(MeshCollisionFragment):
    """``physxSDFMeshCollision:*`` mesh-cooking attributes from `PhysxSDFMeshCollisionAPI`_.

    A single-namespace fragment for the PhysX signed-distance-field cooking schema (PhysX-only
    colliders). The ``sdf`` token is written to ``physics:approximation`` by
    :func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`.

    .. _PhysxSDFMeshCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_s_d_f_mesh_collision_a_p_i.html
    """

    _usd_namespace: ClassVar[str | None] = "physxSDFMeshCollision"
    _usd_applied_schema: ClassVar[str | None] = "PhysxSDFMeshCollisionAPI"

    mesh_approximation_name: str = "sdf"
    """Name of mesh collision approximation method. Default: "sdf"."""

    sdf_margin: float | None = None
    """Margin to increase the size of the SDF relative to the mesh bounding-box diagonal [dimensionless].

    Scale-independent (fraction of the bounding-box diagonal). Default value is 0.01. Range: [0, inf).
    """

    sdf_narrow_band_thickness: float | None = None
    """Size of the narrow band around the mesh surface with high-resolution SDF samples [dimensionless].

    Scale-independent (fraction of the bounding-box diagonal). Default value is 0.01. Range: [0, 1].
    """

    sdf_resolution: int | None = None
    """Uniform SDF sampling resolution (largest AABB extent divided by this value) [dimensionless].

    Default value is 256. Range: (1, inf).
    """

    sdf_subgrid_resolution: int | None = None
    """Subgrid resolution enabling SDF sparsity; ``0`` selects a dense SDF [dimensionless].

    Default value is 6. Range: [0, inf).
    """


@configclass
class PhysxTendonAxisRootCfg(FixedTendonFragment):
    """Whole-tendon attributes from `PhysxTendonAxisRootAPI`_.

    This is a *tune-not-apply* fragment: the source asset owns the
    ``PhysxTendonAxisRootAPI:<instance>`` topology, and this fragment selects existing instances
    to tune. Use :class:`PhysxTendonAxisCfg` for the per-joint-axis properties of the same
    fixed tendon.

    Dispatched via :func:`~isaaclab.sim.schemas.apply_fixed_tendon_properties`.

    .. _PhysxTendonAxisRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_root_a_p_i.html
    """

    _usd_applied_schema: ClassVar[str | None] = "PhysxTendonAxisRootAPI"
    func: Callable | str = "isaaclab_physx.sim.schemas.schemas:_tune_tendon_schema"

    instance_names: str | list[str] | None = None
    """Existing tendon instances to tune; ``None`` selects all root instances."""

    tendon_enabled: bool | None = None
    """Whether to enable or disable the tendon."""

    stiffness: float | None = None
    """Spring stiffness term acting on the tendon's length [N/m]."""

    damping: float | None = None
    """The damping term acting on both the tendon length and the tendon-length limits [N·s/m]."""

    limit_stiffness: float | None = None
    """Limit stiffness term acting on the tendon's length limits [N/m]."""

    offset: float | None = None
    """Length offset term for the tendon [m].

    It defines an amount to be added to the accumulated length computed for the tendon. This allows the application
    to actuate the tendon by shortening or lengthening it.
    """

    rest_length: float | None = None
    """Spring rest length of the tendon [m]."""

    lower_limit: float | None = None
    """Lower limit of the tendon's length [m]."""

    upper_limit: float | None = None
    """Upper limit of the tendon's length [m]."""


@configclass
class PhysxTendonAxisCfg(FixedTendonFragment):
    """Per-joint-axis attributes from `PhysxTendonAxisAPI`_.

    The source asset owns each ``PhysxTendonAxisAPI:<instance>``. This fragment selects existing
    instances on the matched joint prims and tunes their contribution to a fixed tendon. A
    ``PhysxTendonAxisRootAPI`` automatically includes the axis API with the same instance name, so
    this fragment can target both the root joint and AxisAPI-only child joints.

    Dispatched via :func:`~isaaclab.sim.schemas.apply_fixed_tendon_properties`.

    .. _PhysxTendonAxisAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_a_p_i.html
    """

    _usd_applied_schema: ClassVar[str | None] = "PhysxTendonAxisAPI"
    func: Callable | str = "isaaclab_physx.sim.schemas.schemas:_tune_tendon_schema"

    instance_names: str | list[str] | None = None
    """Existing tendon-axis instances to tune; ``None`` selects all axis instances."""

    gearing: list[float] | None = None
    """Joint gearing per entry in :attr:`joint_axis` [unitless or m/deg, depending on joint axis]."""

    force_coefficient: list[float] | None = None
    """Joint force coefficient per entry in :attr:`joint_axis` [unitless or m, depending on joint axis]."""

    joint_axis: list[Literal["transX", "transY", "transZ", "rotX", "rotY", "rotZ"]] | None = None
    """Joint axes corresponding to :attr:`gearing` and :attr:`force_coefficient`."""


@configclass
class PhysxTendonAttachmentRootCfg(SpatialTendonFragment):
    """Whole-tendon attributes from `PhysxTendonAttachmentRootAPI`_.

    A spatial-tendon fragment (see :class:`~isaaclab.sim.schemas.SpatialTendonFragment`) for the
    PhysX spatial-tendon schema. This is a *tune-not-apply* fragment: the source asset owns the
    ``PhysxTendonAttachmentRootAPI:<instance>`` topology, and this fragment selects existing roots
    to tune. Its fields are whole-tendon dynamics; attachment and leaf properties remain
    asset-authored.

    Dispatched via :func:`~isaaclab.sim.schemas.apply_spatial_tendon_properties`.

    .. _PhysxTendonAttachmentRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_attachment_root_a_p_i.html
    """

    _usd_applied_schema: ClassVar[str | None] = "PhysxTendonAttachmentRootAPI"
    func: Callable | str = "isaaclab_physx.sim.schemas.schemas:_tune_tendon_schema"

    instance_names: str | list[str] | None = None
    """Existing spatial-tendon instances to tune; ``None`` selects all attachment roots."""

    tendon_enabled: bool | None = None
    """Whether to enable or disable the tendon."""

    stiffness: float | None = None
    """Spring stiffness term acting on the tendon's length [N/m]."""

    damping: float | None = None
    """The damping term acting on both the tendon length and the tendon-length limits [N·s/m]."""

    limit_stiffness: float | None = None
    """Limit stiffness term acting on the tendon's length limits [N/m]."""

    offset: float | None = None
    """Length offset term for the tendon [m].

    It defines an amount to be added to the accumulated length computed for the tendon. This allows the application
    to actuate the tendon by shortening or lengthening it.
    """
