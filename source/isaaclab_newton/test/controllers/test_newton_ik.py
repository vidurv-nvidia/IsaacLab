# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for NewtonIKController construction, properties, and solve behavior."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest
import warp as wp
from isaaclab_newton.controllers import (
    NewtonIKController,
    NewtonIKControllerCfg,
    build_single_arm_ik_model,
)

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@pytest.fixture(scope="module")
def franka_ik_setup():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    model, info = build_single_arm_ik_model(
        asset_cfg=FRANKA_PANDA_CFG,
        body_name="panda_hand",
        joint_names=["panda_joint.*"],
        device=device,
    )
    return model, info, device


def test_controller_constructs_and_exposes_action_dim(franka_ik_setup):
    model, info, device = franka_ik_setup
    cfg = NewtonIKControllerCfg(command_type="pose", use_relative_mode=True)
    ctrl = NewtonIKController(
        cfg=cfg,
        ik_model=model,
        num_envs=4,
        ee_link_index=info.ee_link_index,
        arm_dof_count=info.arm_dof_count,
        device=device,
    )
    assert ctrl.action_dim == 6
    assert ctrl.num_arm_dofs == info.arm_dof_count
    assert ctrl.num_envs == 4
