# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton inverse-kinematics action term for IsaacLab manager-based envs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers.action_manager import ActionTerm

from isaaclab_newton.assets.articulation import Articulation
from isaaclab_newton.controllers import NewtonIKController, build_single_arm_ik_model

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg

logger = logging.getLogger(__name__)


class NewtonInverseKinematicsAction(ActionTerm):
    """Newton optimization-based IK action term for manager-based envs.

    Owns a :class:`~isaaclab_newton.controllers.NewtonIKController` and a
    dedicated single-arm Newton :class:`~newton.Model` built from the asset's
    USD. Each :meth:`process_actions` call computes EE pose targets in the
    robot-base frame and pushes them into the controller's IK objectives; each
    :meth:`apply_actions` call solves IK across all envs in one batched pass
    and writes the resulting joint targets via
    :meth:`Articulation.set_joint_position_target_index`.
    """

    cfg: "NewtonInverseKinematicsActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "NewtonInverseKinematicsActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)

        # Resolve joint and body indices on the sim asset.
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly 1 body matching '{self.cfg.body_name}', got {body_names}"
            )
        self._body_idx = body_ids[0]

        # Build a single-arm IK Newton model (and validate topology match).
        ik_model, ik_info = build_single_arm_ik_model(
            asset_cfg=self._asset.cfg,
            body_name=self.cfg.body_name,
            joint_names=self.cfg.joint_names,
            device=self.device,
        )
        if ik_info.arm_dof_count != len(self._joint_ids):
            raise ValueError(
                f"IK model arm DOF count ({ik_info.arm_dof_count}) does not match "
                f"sim asset joint count ({len(self._joint_ids)})."
            )
        self._ik_info = ik_info

        # EE offset from cfg.
        if self.cfg.body_offset is not None:
            ee_link_offset = tuple(self.cfg.body_offset.pos)
            ee_link_offset_rot = tuple(self.cfg.body_offset.rot)
        else:
            ee_link_offset = (0.0, 0.0, 0.0)
            ee_link_offset_rot = (0.0, 0.0, 0.0, 1.0)

        self._controller = NewtonIKController(
            cfg=self.cfg.controller,
            ik_model=ik_model,
            num_envs=self.num_envs,
            ee_link_index=ik_info.ee_link_index,
            arm_dof_count=ik_info.arm_dof_count,
            ee_link_offset=ee_link_offset,
            ee_link_offset_rot=ee_link_offset_rot,
            device=self.device,
        )

        # Action tensors.
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Scale: broadcast to (1, action_dim).
        if isinstance(self.cfg.scale, (int, float)):
            self._scale = torch.full((1, self.action_dim), float(self.cfg.scale), device=self.device)
        else:
            self._scale = torch.tensor(self.cfg.scale, device=self.device).reshape(1, -1)
            if self._scale.shape[1] != self.action_dim:
                raise ValueError(
                    f"scale tuple length ({self._scale.shape[1]}) must match action_dim ({self.action_dim})"
                )

        # Body offset tensors for EE pose composition (if set).
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos = None
            self._offset_rot = None

    # ---- properties ----

    @property
    def action_dim(self) -> int:
        return self._controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # ---- operations ----

    def process_actions(self, actions: torch.Tensor) -> None:
        """Scale/clip actions and push EE pose targets into the IK objectives."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale
        ee_pos_b, ee_quat_b = self._compute_ee_pose_in_base_frame()
        self._controller.set_command(self._processed_actions, ee_pos=ee_pos_b, ee_quat=ee_quat_b)

    def apply_actions(self) -> None:
        """Solve IK across all envs in one pass and write joint position targets."""
        joint_pos = wp.to_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        joint_targets = self._controller.compute(joint_pos)
        self._asset.set_joint_position_target_index(target=joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
        self._controller.reset(env_ids=None)

    # ---- helpers ----

    def _compute_ee_pose_in_base_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute EE pose (with optional offset) in the robot-base frame."""
        ee_pos_w = wp.to_torch(self._asset.data.body_pos_w)[:, self._body_idx]
        ee_quat_w = wp.to_torch(self._asset.data.body_quat_w)[:, self._body_idx]
        root_pos_w = wp.to_torch(self._asset.data.root_pos_w)
        root_quat_w = wp.to_torch(self._asset.data.root_quat_w)
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        if self._offset_pos is not None:
            ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pos_b, ee_quat_b, self._offset_pos, self._offset_rot
            )
        return ee_pos_b, ee_quat_b
