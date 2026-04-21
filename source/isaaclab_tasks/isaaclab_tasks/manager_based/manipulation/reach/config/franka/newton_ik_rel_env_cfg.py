# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka reach env using Newton physics + Newton IK relative-pose control.

Train command::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Reach-Franka-Newton-IK-Rel-v0 --num_envs 4096 --headless

Play command::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Reach-Franka-Newton-IK-Rel-Play-v0
"""

from isaaclab_newton.controllers import NewtonIKControllerCfg
from isaaclab_newton.envs.mdp.actions import NewtonInverseKinematicsActionCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaNewtonIKReachEnvCfg(joint_pos_env_cfg.FrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Switch to Newton physics (parent uses PhysX preset by default).
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=50,
                nconmax=20,
                cone="pyramidal",
                integrator="implicitfast",
                impratio=1,
            ),
            num_substeps=1,
            debug_mode=False,
        )

        # Use the high-PD Franka so IK joint targets track well.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Newton IK action term.
        self.actions.arm_action = NewtonInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=NewtonIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                iterations=8,
            ),
            body_offset=NewtonInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaNewtonIKReachEnvCfg_PLAY(FrankaNewtonIKReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
