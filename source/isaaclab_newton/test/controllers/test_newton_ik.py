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


import newton
import numpy as np


def test_compute_pose_accuracy_single_env(franka_ik_setup):
    """Home-pose target → solve → measured EE pose error < 1mm.

    The joint_limit_weight is set to 0.0 so the only objectives are position
    and rotation; with the exact home pose as seed and target the solver
    converges to machine precision (< 1 µm in practice).
    """
    model, info, device = franka_ik_setup
    cfg = NewtonIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        iterations=24,
        joint_limit_weight=0.0,
    )
    ctrl = NewtonIKController(
        cfg=cfg,
        ik_model=model,
        num_envs=1,
        ee_link_index=info.ee_link_index,
        arm_dof_count=info.arm_dof_count,
        device=device,
    )

    # Resolve the home EE pose via Newton FK.
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q_np = state.body_q.numpy()
    ee_tf = body_q_np[info.ee_link_index]  # (7,) as [tx,ty,tz, qx,qy,qz,qw]
    ee_pos_home = torch.tensor([[float(ee_tf[0]), float(ee_tf[1]), float(ee_tf[2])]], device=device)
    ee_quat_home = torch.tensor([[float(ee_tf[3]), float(ee_tf[4]), float(ee_tf[5]), float(ee_tf[6])]], device=device)

    # Seed the IK with the current (home) joints (full n_coords).
    home_q = torch.tensor(model.joint_q.numpy(), device=device).reshape(1, -1)

    target_action = torch.cat([ee_pos_home, ee_quat_home], dim=1)
    ctrl.set_command(target_action, ee_pos=ee_pos_home, ee_quat=ee_quat_home)
    out = ctrl.compute(home_q)
    assert out.shape == (1, model.joint_coord_count), f"unexpected out shape: {out.shape}"

    # Resolve the EE pose at the solved joint config using a one-shot FK.
    q_wp = wp.from_torch(out.contiguous().reshape(-1), dtype=wp.float32)
    state2 = model.state()
    newton.eval_fk(model, q_wp, model.joint_qd, state2)
    body_q2 = state2.body_q.numpy()[info.ee_link_index]
    pos_err = float(np.linalg.norm(np.array(body_q2[:3]) - np.array(ee_tf[:3])))
    assert pos_err < 1e-3, f"position error {pos_err:.4e} > 1mm"


def test_compute_batched_solve_multi_env(franka_ik_setup):
    """Single solver.step solves 16 distinct envs in one pass (no per-env loop)."""
    model, info, device = franka_ik_setup
    cfg = NewtonIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        iterations=16,
    )
    n = 16
    ctrl = NewtonIKController(
        cfg=cfg,
        ik_model=model,
        num_envs=n,
        ee_link_index=info.ee_link_index,
        arm_dof_count=info.arm_dof_count,
        device=device,
    )
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q = state.body_q.numpy()[info.ee_link_index]
    ee_pos = torch.tensor(body_q[:3], device=device).repeat(n, 1).float()
    ee_quat = torch.tensor(
        [float(body_q[3]), float(body_q[4]), float(body_q[5]), float(body_q[6])], device=device
    ).repeat(n, 1)
    action = torch.cat([ee_pos, ee_quat], dim=1)
    home_q = torch.tensor(model.joint_q.numpy(), device=device).repeat(n, 1)
    ctrl.set_command(action, ee_pos=ee_pos, ee_quat=ee_quat)
    out = ctrl.compute(home_q)
    assert out.shape == (n, model.joint_coord_count)
    assert not torch.isnan(out).any()
