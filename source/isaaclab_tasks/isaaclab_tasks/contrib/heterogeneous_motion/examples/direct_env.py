# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .motion_reference import get_combination_ids, reference_joint_positions

if TYPE_CHECKING:
    from .direct_env_cfg import HeterogeneousMotionDirectEnvCfg


class HeterogeneousMotionDirectEnv(DirectRLEnv):
    """Minimal direct environment with clone-combination-selected references."""

    cfg: HeterogeneousMotionDirectEnvCfg

    def __init__(self, cfg: HeterogeneousMotionDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot: Articulation = self.scene["robot"]
        self.combination_ids = get_combination_ids(self.scene)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.robot.data.default_joint_pos.torch + 0.5 * self.actions
        self.robot.set_joint_position_target_index(target=target)

    def _get_observations(self) -> dict:
        reference = self._reference_joint_positions()
        observation = torch.cat(
            (
                self.robot.data.joint_pos.torch - self.robot.data.default_joint_pos.torch,
                self.robot.data.joint_vel.torch,
                reference - self.robot.data.default_joint_pos.torch,
                self.combination_ids.unsqueeze(-1).float(),
            ),
            dim=-1,
        )
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        error = torch.mean(torch.square(self.robot.data.joint_pos.torch - self._reference_joint_positions()), dim=-1)
        return torch.exp(-5.0 * error)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        joint_position = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_velocity = self.robot.data.default_joint_vel.torch[env_ids].clone()

        self.robot.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)

    def _reference_joint_positions(self) -> torch.Tensor:
        phase = self.episode_length_buf * self.step_dt
        return reference_joint_positions(self.robot.data.default_joint_pos.torch, self.combination_ids, phase)
