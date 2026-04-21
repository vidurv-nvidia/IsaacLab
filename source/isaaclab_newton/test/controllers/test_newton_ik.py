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


import torch


def test_set_command_pose_relative_pushes_to_objective(franka_ik_setup):
    """Pose-relative delta should land in pos target buffer after set_command."""
    model, info, device = franka_ik_setup
    cfg = NewtonIKControllerCfg(command_type="pose", use_relative_mode=True)
    ctrl = NewtonIKController(
        cfg=cfg,
        ik_model=model,
        num_envs=2,
        ee_link_index=info.ee_link_index,
        arm_dof_count=info.arm_dof_count,
        device=device,
    )
    # Current EE pose for both envs (identity rotation, origin position).
    torch_device = device if device == "cpu" else device
    ee_pos = torch.zeros(2, 3, device=torch_device)
    ee_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, device=torch_device)
    # Pose-relative action: +X 10 cm translation, no rotation.
    delta = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2, device=torch_device)
    ctrl.set_command(delta, ee_pos=ee_pos, ee_quat=ee_quat)
    target_pos_np = ctrl._target_pos_wp.numpy()
    assert abs(target_pos_np[0][0] - 0.1) < 1e-6, f"expected 0.1, got {target_pos_np[0][0]}"
    assert abs(target_pos_np[1][0] - 0.1) < 1e-6


def test_set_command_pose_absolute_pushes_to_objective(franka_ik_setup):
    """Absolute pose action should land both position and quaternion in target buffers."""
    model, info, device = franka_ik_setup
    cfg = NewtonIKControllerCfg(command_type="pose", use_relative_mode=False)
    ctrl = NewtonIKController(
        cfg=cfg,
        ik_model=model,
        num_envs=2,
        ee_link_index=info.ee_link_index,
        arm_dof_count=info.arm_dof_count,
        device=device,
    )
    torch_device = device if device == "cpu" else device
    target_pos = torch.tensor([[0.3, 0.2, 0.5], [0.4, 0.1, 0.6]], device=torch_device)
    target_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, device=torch_device)
    action = torch.cat([target_pos, target_quat], dim=1)
    ctrl.set_command(action)
    assert abs(ctrl._target_pos_wp.numpy()[0][0] - 0.3) < 1e-6
    assert abs(ctrl._target_pos_wp.numpy()[1][2] - 0.6) < 1e-6


def test_set_command_position_relative(franka_ik_setup):
    model, info, device = franka_ik_setup
    cfg = NewtonIKControllerCfg(command_type="position", use_relative_mode=True)
    ctrl = NewtonIKController(
        cfg=cfg,
        ik_model=model,
        num_envs=1,
        ee_link_index=info.ee_link_index,
        arm_dof_count=info.arm_dof_count,
        device=device,
    )
    torch_device = device if device == "cpu" else device
    ee_pos = torch.tensor([[0.5, 0.0, 0.3]], device=torch_device)
    delta = torch.tensor([[0.05, 0.0, 0.0]], device=torch_device)
    ctrl.set_command(delta, ee_pos=ee_pos)
    assert abs(ctrl._target_pos_wp.numpy()[0][0] - 0.55) < 1e-6


def test_set_command_position_absolute(franka_ik_setup):
    model, info, device = franka_ik_setup
    cfg = NewtonIKControllerCfg(command_type="position", use_relative_mode=False)
    ctrl = NewtonIKController(
        cfg=cfg,
        ik_model=model,
        num_envs=1,
        ee_link_index=info.ee_link_index,
        arm_dof_count=info.arm_dof_count,
        device=device,
    )
    torch_device = device if device == "cpu" else device
    target_pos = torch.tensor([[0.6, 0.1, 0.4]], device=torch_device)
    ctrl.set_command(target_pos)
    assert abs(ctrl._target_pos_wp.numpy()[0][0] - 0.6) < 1e-6
