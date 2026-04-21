# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from isaaclab_newton.controllers import NewtonIKControllerCfg
from isaaclab_newton.envs.mdp.actions import NewtonInverseKinematicsActionCfg


def test_action_cfg_defaults_are_sane():
    cfg = NewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=NewtonIKControllerCfg(),
    )
    assert cfg.asset_name == "robot"
    assert cfg.body_name == "panda_hand"
    assert cfg.scale == 1.0
    assert cfg.body_offset is None
    assert cfg.controller.iterations == 8
