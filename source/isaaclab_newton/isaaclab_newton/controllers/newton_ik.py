# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched Newton-native inverse-kinematics controller for IsaacLab."""

from __future__ import annotations

import newton
import newton.ik as ik
import numpy as np
import torch
import warp as wp

from .newton_ik_cfg import NewtonIKControllerCfg


class NewtonIKController:
    """Batched optimization-based IK controller backed by :mod:`newton.ik`.

    Each :meth:`compute` call runs ``cfg.iterations`` optimizer steps for all
    ``num_envs`` problems in one fused Warp kernel launch. Targets are
    end-effector pose(s) in the **robot-base frame**; the IK model is built
    with its base at the world origin so no per-env frame correction is needed
    inside this class — the action term is responsible for converting world
    poses into base-frame targets before calling :meth:`set_command`.

    Args:
        cfg: Controller configuration.
        ik_model: Single-articulation Newton model (built via
            :func:`~isaaclab_newton.controllers.build_single_arm_ik_model`).
        num_envs: Number of environments solved together per :meth:`compute`.
        ee_link_index: Index of the EE body in ``ik_model.body_q``.
        arm_dof_count: Number of arm-controlled DOFs (IK-relevant subset).
            When provided, :attr:`num_arm_dofs` returns this value. When
            ``None``, falls back to ``ik_model.joint_coord_count``.
        ee_link_offset: Translation [m] of the constrained point in the EE
            link's local frame.
        ee_link_offset_rot: Quaternion ``(x, y, z, w)`` of the constrained
            frame in the EE link's local frame.
        device: Warp device string.
    """

    def __init__(
        self,
        cfg: NewtonIKControllerCfg,
        ik_model: newton.Model,
        num_envs: int,
        ee_link_index: int,
        arm_dof_count: int | None = None,
        ee_link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ee_link_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        device: str = "cuda:0",
    ) -> None:
        # Validate (calls validate_config hook via @configclass).
        cfg.validate_config()
        self.cfg = cfg
        self.model = ik_model
        self.num_envs = num_envs
        self.device = device

        self._n_coords = ik_model.joint_coord_count
        self._arm_dof_count = arm_dof_count if arm_dof_count is not None else self._n_coords

        # Per-env target buffers (allocated empty; updated by set_command).
        self._target_pos_wp = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        # Initialize quat targets to identity so quat residual is well-defined on first solve.
        ident_np = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_envs, 1))
        self._target_rot_wp = wp.array(ident_np, dtype=wp.vec4, device=device)

        # Objectives.
        self._pos_obj = ik.IKObjectivePosition(
            link_index=ee_link_index,
            link_offset=wp.vec3(*ee_link_offset),
            target_positions=self._target_pos_wp,
            weight=cfg.position_weight,
        )
        self._rot_obj = ik.IKObjectiveRotation(
            link_index=ee_link_index,
            link_offset_rotation=wp.quat(*ee_link_offset_rot),
            target_rotations=self._target_rot_wp,
            weight=cfg.rotation_weight,
        )
        self._limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=ik_model.joint_limit_lower,
            joint_limit_upper=ik_model.joint_limit_upper,
            weight=cfg.joint_limit_weight,
        )

        objectives: list[ik.IKObjective] = [self._pos_obj, self._rot_obj, self._limit_obj]
        if cfg.command_type == "position":
            objectives = [self._pos_obj, self._limit_obj]

        self._solver = ik.IKSolver(
            model=ik_model,
            n_problems=num_envs,
            objectives=objectives,
            optimizer=cfg.optimizer,
            jacobian_mode=cfg.jacobian_mode,
            sampler=cfg.sampler,
            n_seeds=cfg.n_seeds,
            noise_std=cfg.noise_std,
            rng_seed=cfg.rng_seed,
            lambda_initial=cfg.lambda_initial,
            lambda_factor=cfg.lambda_factor,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
            rho_min=cfg.rho_min,
            history_len=cfg.history_len,
            h0_scale=cfg.h0_scale,
        )

        self._joint_q_in_wp = wp.zeros((num_envs, self._n_coords), dtype=wp.float32, device=device)
        self._joint_q_out_wp = wp.zeros((num_envs, self._n_coords), dtype=wp.float32, device=device)
        self._previous_solution_wp: wp.array | None = None
        if cfg.seed_source == "previous_solution":
            self._previous_solution_wp = wp.zeros_like(self._joint_q_in_wp)

    @property
    def action_dim(self) -> int:
        """Action dimension implied by ``cfg`` (3, 6, or 7)."""
        return self.cfg.action_dim()

    @property
    def num_arm_dofs(self) -> int:
        """Number of arm-controlled joint DOFs.

        When ``arm_dof_count`` is passed to the constructor, this returns that
        value (the IK-controlled subset). Otherwise falls back to the full
        ``joint_coord_count`` of the IK model.
        """
        return self._arm_dof_count

    def set_command(
        self,
        command: torch.Tensor,
        ee_pos: torch.Tensor | None = None,
        ee_quat: torch.Tensor | None = None,
    ) -> None:
        """Set the EE pose target(s) for the next :meth:`compute` call.

        Args:
            command: Target command tensor, shape ``[num_envs, action_dim]``.
                Interpretation depends on :attr:`cfg.command_type` and
                :attr:`cfg.use_relative_mode`. For ``pose`` + relative this is
                ``(dx, dy, dz, drx, dry, drz)``.
            ee_pos: Current EE position [m], shape ``[num_envs, 3]``. Required
                for any relative mode.
            ee_quat: Current EE orientation ``(x, y, z, w)``, shape
                ``[num_envs, 4]``. Required for any pose mode with relative.
        """
        from isaaclab.utils.math import apply_delta_pose

        if self.cfg.command_type == "pose":
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError("ee_pos and ee_quat are required for relative pose mode")
                target_pos, target_quat = apply_delta_pose(ee_pos, ee_quat, command)
            else:
                target_pos = command[:, 0:3]
                target_quat = command[:, 3:7]
            self._write_pos_target(target_pos)
            self._write_quat_target(target_quat)
        else:  # position-only
            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("ee_pos is required for position_rel mode")
                target_pos = ee_pos + command
            else:
                target_pos = command
            self._write_pos_target(target_pos)

    def _write_pos_target(self, target_pos: torch.Tensor) -> None:
        """Copy a torch ``(N, 3)`` position tensor into the Warp target_positions buffer."""
        target_pos_wp = wp.from_torch(target_pos.contiguous(), dtype=wp.vec3)
        wp.copy(self._target_pos_wp, target_pos_wp)

    def _write_quat_target(self, target_quat: torch.Tensor) -> None:
        """Copy a torch ``(N, 4)`` quaternion tensor into the Warp target_rotations buffer."""
        target_quat_wp = wp.from_torch(target_quat.contiguous(), dtype=wp.vec4)
        wp.copy(self._target_rot_wp, target_quat_wp)

    def compute(self, joint_q_in: torch.Tensor) -> torch.Tensor:
        """Solve IK for all envs in one batched pass.

        Args:
            joint_q_in: Current joint positions [m or rad] for the full IK
                model, shape ``[num_envs, joint_coord_count]``. The caller
                (typically the action term) is responsible for providing
                complete joint coords — including non-arm joints such as
                gripper fingers, which the solver leaves unchanged because
                they have zero position/rotation Jacobian contribution.

        Returns:
            Target joint positions [m or rad], shape
            ``[num_envs, joint_coord_count]``.
        """
        if joint_q_in.shape != (self.num_envs, self._n_coords):
            raise ValueError(
                f"joint_q_in shape {tuple(joint_q_in.shape)} does not match "
                f"(num_envs={self.num_envs}, joint_coord_count={self._n_coords})"
            )

        if self.cfg.seed_source == "sim_joint_pos":
            seed_wp = wp.from_torch(joint_q_in.contiguous(), dtype=wp.float32).reshape((self.num_envs, self._n_coords))
            wp.copy(self._joint_q_in_wp, seed_wp)
        elif self.cfg.seed_source == "previous_solution":
            assert self._previous_solution_wp is not None
            wp.copy(self._joint_q_in_wp, self._previous_solution_wp)
        elif self.cfg.seed_source == "default_pose":
            default_q = self.model.joint_q.numpy()  # full n_coords
            tile = torch.tensor(default_q, device=joint_q_in.device).repeat(self.num_envs, 1)
            wp.copy(
                self._joint_q_in_wp,
                wp.from_torch(tile.contiguous(), dtype=wp.float32).reshape((self.num_envs, self._n_coords)),
            )
        else:
            raise ValueError(f"unknown seed_source: {self.cfg.seed_source}")

        self._solver.step(
            self._joint_q_in_wp,
            self._joint_q_out_wp,
            iterations=self.cfg.iterations,
        )

        if self._previous_solution_wp is not None:
            wp.copy(self._previous_solution_wp, self._joint_q_out_wp)

        return wp.to_torch(self._joint_q_out_wp)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal solver/seed state.

        Args:
            env_ids: Currently unused (the underlying Newton IK solver has no
                per-env state worth preserving across resets). Present for
                parity with the IsaacLab :class:`ActionTerm` reset signature.
        """
        del env_ids  # unused for now
        self._solver.reset()
        if self._previous_solution_wp is not None:
            self._previous_solution_wp.zero_()
