# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 flat-terrain velocity tracking with Newton's sparse BDF2 VBD solver."""

from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg

from isaaclab.utils import configclass

from .flat_env_cfg import G1FlatEnvCfg


@configclass
class _G1VBDSolverCfg(VBDSolverCfg):
    """Rigid articulation options supplied by the pinned Newton VBD development revision."""

    rigid_articulation_solve: str = "block_sparse_joints"
    rigid_time_integration: str = "bdf2"
    rigid_articulation_relaxation: float = 0.65
    rigid_joint_linear_kd: float = 500.0
    rigid_joint_angular_kd: float = 500.0
    rigid_axis_support_armature: bool = True
    rigid_limit_rho_floor: bool = True


@configclass
class G1FlatVBDEnvCfg(G1FlatEnvCfg):
    """Use VBD with the existing G1 asset, internal drives, rewards, and 50 Hz policy."""

    def __post_init__(self):
        super().__post_init__()

        newton_mjwarp = self.sim.physics.newton_mjwarp
        self.sim.physics.default = NewtonCfg(
            solver_cfg=_G1VBDSolverCfg(
                iterations=7,
                rigid_compliant_alm=True,
                rigid_joint_linear_ke=1.0e6,
                rigid_joint_angular_ke=1.0e6,
                rigid_body_contact_buffer_size=256,
            ),
            collision_cfg=newton_mjwarp.collision_cfg,
            default_shape_cfg=newton_mjwarp.default_shape_cfg,
            num_substeps=1,
            collision_decimation=0,
        )
        # Preserve the original 20 ms action/observation interval with 4 ms physics.
        self.sim.dt = 0.004
        self.decimation = 5
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
