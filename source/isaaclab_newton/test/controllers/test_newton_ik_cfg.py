# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for NewtonIKControllerCfg."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest

from isaaclab_newton.controllers import NewtonIKControllerCfg


def test_default_cfg_is_pose_relative_lm_analytic_warm():
    cfg = NewtonIKControllerCfg()
    assert cfg.command_type == "pose"
    assert cfg.use_relative_mode is True
    assert cfg.optimizer == "lm"
    assert cfg.jacobian_mode == "analytic"
    assert cfg.iterations == 8
    assert cfg.sampler == "none"
    assert cfg.n_seeds == 1
    assert cfg.seed_source == "sim_joint_pos"


def test_action_dim_helper():
    # 3-D for position-only
    cfg = NewtonIKControllerCfg(command_type="position", use_relative_mode=False)
    assert cfg.action_dim() == 3
    # 6-D for pose-relative
    cfg = NewtonIKControllerCfg(command_type="pose", use_relative_mode=True)
    assert cfg.action_dim() == 6
    # 7-D for absolute pose
    cfg = NewtonIKControllerCfg(command_type="pose", use_relative_mode=False)
    assert cfg.action_dim() == 7


def test_invalid_optimizer_raises():
    with pytest.raises(ValueError):
        NewtonIKControllerCfg(optimizer="bogus").validate_config()
