# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

from .newton_collision_cfg import NewtonCollisionPipelineCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class HydroelasticCfg:
    """Configuration for hydroelastic contact handling.

    Hydroelastic contacts use SDF overlap between shape pairs to compute distributed
    contact surfaces with area-weighted forces via marching cubes. This requires SDF
    to be enabled on the parent :class:`SDFCfg`.

    Both shapes in a colliding pair must have the hydroelastic flag for hydroelastic
    contacts to be generated. Shapes that only have SDF (but not hydroelastic) will
    fall back to standard point contacts.

    .. note::
        Hydroelastic contacts require the unified collision pipeline, which is used
        when ``use_mujoco_contacts=False`` on :class:`MJWarpSolverCfg`, or when using
        a non-MuJoCo solver (XPBD, Featherstone).
    """

    k_hydro: float = 1e10
    """Hydroelastic stiffness coefficient [Pa] applied to shapes.

    Controls the compliance of the hydroelastic contact surface. Higher values produce
    stiffer contacts.
    """

    shape_patterns: list[str] | None = None
    """Regex patterns to select which shapes get hydroelastic contacts.

    If None, all shapes that have SDF enabled will also get hydroelastic contacts.
    If provided, only shapes whose label (USD prim path) matches at least one pattern
    will have the ``HYDROELASTIC`` flag set.

    Example: ``[".*Gear.*", ".*gear.*"]``
    """

    reduce_contacts: bool = True
    """Whether to reduce contacts to a smaller representative set per shape pair."""

    output_contact_surface: bool = False
    """Whether to output hydroelastic contact surface vertices for visualization."""

    normal_matching: bool = True
    """Whether to adjust reduced contact normals so their net force direction matches
    the unreduced reference. Only active when :attr:`reduce_contacts` is True."""

    moment_matching: bool = False
    """Whether to adjust reduced contact friction coefficients so their net maximum
    moment matches the unreduced reference. Only active when :attr:`reduce_contacts`
    is True."""

    margin_contact_area: float = 1e-2
    """Contact area [m^2] used for non-penetrating contacts at the margin."""

    buffer_mult_broad: int = 1
    """Multiplier for the preallocated broadphase buffer. Increase if a broadphase
    overflow warning is issued."""

    buffer_mult_iso: int = 1
    """Multiplier for preallocated iso-surface extraction buffers. Increase if an
    iso buffer overflow warning is issued."""

    buffer_mult_contact: int = 1
    """Multiplier for the preallocated face contact buffer. Increase if a face
    contact overflow warning is issued."""

    grid_size: int = 256 * 8 * 128
    """Grid size for contact handling. Can be tuned for performance."""


@configclass
class SDFCfg:
    """Configuration for SDF (Signed Distance Field) collision on Newton meshes.

    When provided as ``sdf_cfg`` on :class:`NewtonCfg`, mesh collision shapes loaded
    from USD will have SDF-based collision enabled via Newton's ``mesh.build_sdf()``
    API at simulation start. At least one of ``max_resolution`` or ``target_voxel_size``
    must be set.

    Regex patterns in ``shape_patterns`` and ``pattern_resolutions`` allow selective
    SDF application and per-shape resolution tuning.
    """

    max_resolution: int | None = None
    """Maximum dimension [voxels] for sparse SDF grid (must be divisible by 8).

    If set, mesh collision shapes loaded from USD will have SDF-based collision enabled.
    Typical values: 128, 256, 512.
    """

    narrow_band_range: tuple[float, float] = (-0.1, 0.1)
    """Narrow band distance range (inner, outer) [m] for SDF computation."""

    target_voxel_size: float | None = None
    """Target voxel size [m] for sparse SDF grid.

    If provided, takes precedence over :attr:`max_resolution`.
    """

    margin: float | None = None
    """Collision margin [m] for SDF shapes. If None, uses the builder's default."""

    body_patterns: list[str] | None = None
    """List of regex patterns to match body labels (USD prim paths) for SDF.

    If None, no body-level filtering is applied. If provided, bodies whose label
    matches at least one pattern will have SDF applied to all their mesh shapes.
    Example: ``[".*elbow.*", ".*wrist.*"]``
    """

    shape_patterns: list[str] | None = None
    """List of regex patterns to match shape labels (USD prim paths) for SDF.

    If None, no shape-level filtering is applied. If provided, only shapes whose
    label matches at least one pattern get SDF.
    Example: ``[".*Gear.*", ".*gear.*"]``

    .. note::
        At least one of :attr:`body_patterns` or :attr:`shape_patterns` must be set
        for SDF to be applied. If both are None, no shapes will receive SDF.
    """

    use_visual_meshes: bool = False
    """Whether to create collision shapes from visual meshes for bodies that have
    no collision geometry. When False (default), only existing collision meshes
    are patched with SDF. When True, bodies that match the configured patterns
    but lack collision shapes will get a new collision shape created from their
    first visual mesh.
    """

    pattern_resolutions: dict[str, int] | None = None
    """Per-pattern SDF resolution overrides.

    Maps regex pattern to max_resolution for matching shapes. Shapes not matching any
    pattern here use the global ``max_resolution``. First matching pattern wins.
    Example: ``{".*elbow.*": 128, ".*power_supply.*": 512}``
    """

    hydroelastic_cfg: HydroelasticCfg | None = None
    """Hydroelastic contact configuration.

    If None (default), hydroelastic contacts are disabled and standard point contacts
    are used. When set, shapes matching the SDF patterns (or the hydroelastic-specific
    ``shape_patterns``) will have the ``HYDROELASTIC`` flag enabled and use distributed
    surface contacts computed via SDF overlap.
    """


@configclass
class NewtonSolverCfg:
    """Configuration for Newton solver-related parameters.

    These parameters are used to configure the Newton solver. For more information, see the `Newton documentation`_.

    Subclasses set :attr:`class_type` to their matching :class:`NewtonManager`
    subclass; :class:`NewtonCfg` propagates that to its own
    :attr:`NewtonCfg.class_type` in :meth:`NewtonCfg.__post_init__` so that
    ``SimulationContext`` resolves the correct manager via the existing
    dispatch path.

    .. _Newton documentation: https://newton.readthedocs.io/en/latest/
    """

    class_type: type[NewtonManager] | str = "{DIR}.newton_manager:NewtonManager"
    """Manager class for this solver.

    Default points at the abstract :class:`NewtonManager`; concrete subclasses
    override it.
    """

    solver_type: str = "None"
    """Solver type metadata (deprecated).

    .. deprecated::
        Manager dispatch is now driven by :attr:`class_type`; this field is
        retained as metadata for logging and debugging only.  Do not branch on
        ``solver_type`` in new code.
    """


@configclass
class NewtonShapeCfg:
    """Default per-shape collision properties applied to all shapes in a Newton scene.

    Mirrors Newton's :attr:`ModelBuilder.default_shape_cfg`. Only fields Isaac
    Lab actually overrides are declared here; unspecified fields keep Newton's
    upstream default. The struct is forwarded onto Newton's upstream
    ``ShapeConfig`` via :func:`~isaaclab.utils.checked_apply` at builder
    construction.
    """

    margin: float = 0.0
    """Default per-shape collision margin [m].

    A nonzero margin (e.g. ``0.01``) is required for stable contact on
    triangle-mesh terrain — without it, lightweight robots fail to learn
    rough-terrain locomotion on Newton. Newton's upstream default is ``0.0``.
    """

    gap: float = 0.01
    """Default per-shape contact gap [m]. Newton's upstream default is ``None``."""


@configclass
class NewtonCfg(PhysicsCfg):
    """Configuration for Newton physics manager.

    This configuration includes Newton-specific simulation settings and solver configuration.

    The active :class:`NewtonManager` subclass is determined by
    :attr:`solver_cfg.class_type`, which :meth:`__post_init__` propagates to
    :attr:`class_type` so that ``SimulationContext`` resolves the right
    manager subclass automatically.  User code keeps the existing two-level
    shape ``NewtonCfg(solver_cfg=...)`` and does not need to set
    :attr:`class_type` explicitly.
    """

    class_type: type[NewtonManager] | str | None = None
    """The class type of the :class:`NewtonManager`.

    Auto-set in :meth:`__post_init__` from :attr:`solver_cfg.class_type`.
    Users normally do not set this directly.
    """

    num_substeps: int = 1
    """Number of substeps to use for the solver."""

    debug_mode: bool = False
    """Whether to enable debug mode for the solver."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graphing when simulating.

    If set to False, the simulation performance will be severely degraded.
    """

    solver_cfg: NewtonSolverCfg | None = None
    """Solver configuration. If None (default), MJWarpSolverCfg is used by default."""

    collision_cfg: NewtonCollisionPipelineCfg | None = None
    """Newton collision pipeline configuration.

    Controls how Newton's :class:`CollisionPipeline` is configured when it is active.
    The pipeline is active when the solver delegates collision detection to Newton:

    - :class:`MJWarpSolverCfg` with ``use_mujoco_contacts=False``,
    - :class:`KaminoSolverCfg` with ``use_collision_detector=False``,
    - :class:`XPBDSolverCfg` (always),
    - :class:`FeatherstoneSolverCfg` (always).

    If ``None`` (default), a pipeline with ``broad_phase="explicit"`` is created
    automatically.  Set this to a :class:`NewtonCollisionPipelineCfg` to customize
    parameters such as broad phase algorithm, contact limits, or hydroelastic mode.

    .. note::
        Setting this while ``MJWarpSolverCfg.use_mujoco_contacts=True`` raises
        :class:`ValueError`.  When ``KaminoSolverCfg.use_collision_detector=True``,
        the field is ignored because Kamino's internal detector handles contacts.
    """

    sdf_cfg: SDFCfg | None = None
    """SDF collision configuration. If None (default), SDF is disabled."""

    default_shape_cfg: NewtonShapeCfg = NewtonShapeCfg()
    """Default per-shape collision properties applied to every shape in the scene.

    Forwarded to Newton's :attr:`ModelBuilder.default_shape_cfg` at builder
    construction via :func:`~isaaclab.utils.checked_apply`. See
    :class:`NewtonShapeCfg` for the declared fields.
    """

    def __post_init__(self):
        # NewtonCfg.class_type is auto-derived from solver_cfg.class_type.
        # Refuse a user-set value: setting both is ambiguous and was
        # previously silently overwritten.
        if self.class_type is not None:
            raise TypeError("Cannot manually set NewtonCfg.class_type; it is auto-derived from solver_cfg.class_type.")
        if self.solver_cfg is None:
            from .mjwarp_manager_cfg import MJWarpSolverCfg

            self.solver_cfg = MJWarpSolverCfg()
        self.class_type = self.solver_cfg.class_type
