# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym smoke test for the Franka reach Newton-IK relative-pose env."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  registers gym envs
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def test_gym_make_franka_newton_ik_rel():
    """Create env via gym.make, reset, step 20 times with small random actions, assert no NaN."""
    env_cfg = parse_env_cfg("Isaac-Reach-Franka-Newton-IK-Rel-v0", num_envs=4)
    env = gym.make("Isaac-Reach-Franka-Newton-IK-Rel-v0", cfg=env_cfg)
    try:
        obs, _ = env.reset()
        assert "policy" in obs, f"expected 'policy' key in obs, got {list(obs.keys())}"
        torch.manual_seed(0)
        for _ in range(20):
            action = (torch.rand(4, 6, device=env.unwrapped.device) - 0.5) * 0.02  # ±1cm
            obs, _, _, _, _ = env.step(action)
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                assert not torch.isnan(val).any(), f"NaN in observation '{key}'"
    finally:
        env.close()
