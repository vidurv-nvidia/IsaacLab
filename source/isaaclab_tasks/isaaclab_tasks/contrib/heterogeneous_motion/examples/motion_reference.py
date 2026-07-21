# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.scene import InteractiveScene


def get_combination_ids(scene: InteractiveScene) -> torch.Tensor:
    """Derive each environment's effective combination ID from the clone mask."""
    plan = scene.clone_plan
    if plan is None or plan.clone_mask.shape[0] == 0:
        raise RuntimeError("The scene did not produce a clone mask.")
    _, combination_ids = torch.unique(plan.clone_mask.T, dim=0, return_inverse=True)
    return combination_ids


def reference_joint_positions(
    default_joint_positions: torch.Tensor, combination_ids: torch.Tensor, phase: torch.Tensor
) -> torch.Tensor:
    """Generate two small procedural references selected by clone combination ID."""
    target = default_joint_positions.clone()
    first_motion = combination_ids == 0
    target[:, 0] += torch.where(first_motion, 0.25 * torch.sin(phase), 0.15 * torch.sin(2.0 * phase))
    target[:, 1] += torch.where(first_motion, -0.25 * torch.sin(phase), 0.25 * torch.cos(2.0 * phase))
    return target


def manager_reference_joint_positions(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return the combination-selected joint reference for a manager-based environment."""
    robot: Articulation = env.scene[asset_cfg.name]
    phase = env.episode_length_buf * env.step_dt
    return reference_joint_positions(robot.data.default_joint_pos.torch, _manager_combination_ids(env), phase)


def manager_combination_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the clone combination ID as a one-column policy observation."""
    return _manager_combination_ids(env).unsqueeze(-1).float()


def manager_motion_tracking_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the robot for tracking its combination-selected joint reference."""
    robot: Articulation = env.scene[asset_cfg.name]
    target = manager_reference_joint_positions(env, asset_cfg)
    error = torch.mean(torch.square(robot.data.joint_pos.torch - target), dim=-1)
    return torch.exp(-5.0 * error)


def _manager_combination_ids(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Derive and cache combination IDs once for manager terms."""
    cache_name = "_heterogeneous_motion_combination_ids"
    combination_ids = getattr(env, cache_name, None)
    if combination_ids is None:
        combination_ids = get_combination_ids(env.scene)
        setattr(env, cache_name, combination_ids)
    return combination_ids
