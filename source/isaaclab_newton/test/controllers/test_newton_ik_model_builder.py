# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the single-arm IK Newton Model builder."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest
import warp as wp
from isaaclab_newton.controllers.newton_ik_model_builder import (
    IKModelInfo,
    build_single_arm_ik_model,
)

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


def test_build_franka_ik_model_returns_single_articulation():
    model, info = build_single_arm_ik_model(
        asset_cfg=FRANKA_PANDA_CFG,
        body_name="panda_hand",
        joint_names=["panda_joint.*"],
        device="cuda:0" if wp.is_cuda_available() else "cpu",
    )
    # Newton may produce > 1 articulation for instanceable USDs (e.g. the
    # rootJoint fixed joint creates a separate one-joint articulation).
    assert model.articulation_count >= 1, f"expected at least 1 articulation, got {model.articulation_count}"
    assert isinstance(info, IKModelInfo)
    assert info.arm_dof_count == 7  # Franka has 7 arm DOFs (panda_joint1..panda_joint7)
    assert info.ee_link_index >= 0
    assert len(info.ik_joint_names) == 7
    assert all(n.startswith("panda_joint") for n in info.ik_joint_names)
    assert len(info.sim_to_ik_joint_perm) == 7


def test_build_raises_on_unknown_body():
    with pytest.raises(ValueError, match="not found"):
        build_single_arm_ik_model(
            asset_cfg=FRANKA_PANDA_CFG,
            body_name="this_body_does_not_exist",
            joint_names=["panda_joint.*"],
            device="cuda:0" if wp.is_cuda_available() else "cpu",
        )
