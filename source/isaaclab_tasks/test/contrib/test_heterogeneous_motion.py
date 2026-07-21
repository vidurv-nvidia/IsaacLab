# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-backed validation for the heterogeneous maze demonstrator."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.heterogeneous_motion.validation import TASK_ID, validate_maze
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("num_envs", (2, 4))
def test_heterogeneous_maze_validates_layout_motion_routing(num_envs: int):
    """Run zero/random action, reset, grouping, routing, and completion checks."""
    env_cfg = parse_env_cfg(TASK_ID, device="cuda", num_envs=num_envs)
    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        counts = validate_maze(env, steps=100)
        assert sum(counts.values()) == num_envs
    finally:
        env.close()
