# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from .direct_env import HeterogeneousMotionDirectEnv


class HeterogeneousUnequalAssetsDirectEnv(HeterogeneousMotionDirectEnv):
    """Direct task that leaves immutable optional kinematic assets untouched during reset."""

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Optional rigid-object views contain only the environments where that asset
        # exists, so global environment indices cannot be forwarded to scene.reset().
        self.robot.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        joint_position = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_velocity = self.robot.data.default_joint_vel.torch[env_ids].clone()

        self.robot.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
