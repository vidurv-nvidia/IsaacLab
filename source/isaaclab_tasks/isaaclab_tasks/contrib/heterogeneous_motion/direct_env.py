# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv

from .layouts import LAYOUTS, resolve_layout_ids
from .motion import load_motion_file

if TYPE_CHECKING:
    from .direct_env_cfg import HeterogeneousMazeDirectEnvCfg


class HeterogeneousMazeDirectEnv(DirectRLEnv):
    """Planar path-following demo with a motion file selected by physical layout."""

    cfg: HeterogeneousMazeDirectEnvCfg

    def __init__(self, cfg: HeterogeneousMazeDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.agent: RigidObject = self.scene["agent"]
        self.actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._velocity_command = torch.zeros((self.num_envs, 6), device=self.device)

        self.combination_rows, self.combination_ids, self.layout_ids = resolve_layout_ids(self.scene)
        self.motion_files = tuple(layout.motion_file for layout in LAYOUTS)
        self._load_motion_paths()

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions[:] = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        self._velocity_command.zero_()
        self._velocity_command[:, :2] = self.cfg.max_speed * self.actions
        self.agent.write_root_velocity_to_sim_index(root_velocity=self._velocity_command)

    def _get_observations(self) -> dict:
        position = self._agent_position()
        velocity = self.agent.data.root_lin_vel_w.torch[:, :2]
        reference_velocity, _ = self._reference_velocity(position)
        return {"policy": torch.cat((position, velocity, reference_velocity), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        position = self._agent_position()
        velocity = self.agent.data.root_lin_vel_w.torch[:, :2]
        reference_velocity, path_distance_squared = self._reference_velocity(position)
        velocity_error_squared = torch.sum(torch.square(velocity - reference_velocity), dim=-1)
        return -velocity_error_squared - 0.1 * path_distance_squared

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        position = self._agent_position()
        goal_distance = torch.linalg.vector_norm(position - self._goal_positions(), dim=-1)
        terminated = goal_distance < self.cfg.goal_tolerance
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Optional wall views contain compact indices, so reset only the shared agent.
        self.agent.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        root_pose = self.agent.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_velocity = self.agent.data.default_root_vel.torch[env_ids].clone()
        self.agent.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.agent.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    def _load_motion_paths(self):
        paths = [load_motion_file(path, self.device) for path in self.motion_files]
        self._motion_lengths = torch.tensor([path.shape[0] for path in paths], device=self.device)
        self._motion_paths = torch.empty((len(paths), int(self._motion_lengths.max().item()), 2), device=self.device)
        for index, path in enumerate(paths):
            self._motion_paths[index] = path[-1]
            self._motion_paths[index, : path.shape[0]] = path

    def _agent_position(self) -> torch.Tensor:
        return self.agent.data.root_pos_w.torch[:, :2] - self.scene.env_origins[:, :2]

    def _goal_positions(self) -> torch.Tensor:
        lengths = self._motion_lengths[self.layout_ids]
        return self._motion_paths[self.layout_ids, lengths - 1]

    def _reference_velocity(self, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        paths = self._motion_paths[self.layout_ids]
        distance_squared = torch.sum(torch.square(paths - position.unsqueeze(1)), dim=-1)
        nearest_indices = torch.argmin(distance_squared, dim=1)
        lengths = self._motion_lengths[self.layout_ids]
        lookahead_indices = torch.minimum(nearest_indices + self.cfg.lookahead_points, lengths - 1)
        targets = paths[torch.arange(self.num_envs, device=self.device), lookahead_indices]
        direction = torch.nn.functional.normalize(targets - position, dim=-1, eps=1.0e-6)
        nearest_distance_squared = distance_squared[
            torch.arange(self.num_envs, device=self.device),
            nearest_indices,
        ]
        return self.cfg.max_speed * direction, nearest_distance_squared
