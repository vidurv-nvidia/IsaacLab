# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`~isaaclab_newton.controllers.NewtonIKController`."""

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass


@configclass
class NewtonIKControllerCfg:
    """Configuration for the Newton optimization-based IK controller.

    The defaults match the Standard tracking setup: relative-pose 6-D action,
    Levenberg-Marquardt optimizer with analytic Jacobians, 8 iterations per
    step, no multi-seed sampling, and warm-starting from the current sim
    joint positions.
    """

    command_type: Literal["position", "pose"] = "pose"
    """Whether the action command is position-only or full pose."""

    use_relative_mode: bool = True
    """If ``True``, the command is interpreted as a delta applied to the
    current end-effector pose."""

    # --- Solver ---
    optimizer: Literal["lm", "lbfgs"] = "lm"
    """Newton IK optimizer backend."""

    jacobian_mode: Literal["analytic", "autodiff", "mixed"] = "analytic"
    """Jacobian computation backend."""

    iterations: int = 8
    """Number of optimizer iterations per :meth:`compute` call."""

    # --- Sampling (only used when n_seeds > 1) ---
    sampler: Literal["none", "gauss", "roberts", "uniform"] = "none"
    n_seeds: int = 1
    noise_std: float = 0.1
    rng_seed: int = 12345

    # --- Objective weights ---
    position_weight: float = 1.0
    rotation_weight: float = 1.0
    joint_limit_weight: float = 0.1

    # --- LM-specific (ignored when ``optimizer == "lbfgs"``) ---
    lambda_initial: float = 0.1
    lambda_factor: float = 2.0
    lambda_min: float = 1e-5
    lambda_max: float = 1e10
    rho_min: float = 1e-3

    # --- L-BFGS-specific (ignored when ``optimizer == "lm"``) ---
    history_len: int = 10
    h0_scale: float = 1.0

    # --- Warm-start ---
    seed_source: Literal["sim_joint_pos", "previous_solution", "default_pose"] = "sim_joint_pos"
    """Source of the seed joint positions for each :meth:`compute` call."""

    def action_dim(self) -> int:
        """Return the action dimension implied by this configuration.

        Returns:
            ``3`` for position-only, ``6`` for pose-relative, or ``7`` for
            absolute pose.
        """
        if self.command_type == "position":
            return 3
        if self.use_relative_mode:
            return 6
        return 7

    def validate_config(self) -> None:
        """Validate enum-like fields beyond what :func:`Literal` enforces.

        This method is automatically called by :func:`~isaaclab.utils.configclass`
        after the standard missing-field check.

        Raises:
            ValueError: If any enum-like field has an unsupported value or a
                numeric field is out of range.
        """
        if self.optimizer not in ("lm", "lbfgs"):
            raise ValueError(f"optimizer must be 'lm' or 'lbfgs', got {self.optimizer!r}")
        if self.jacobian_mode not in ("analytic", "autodiff", "mixed"):
            raise ValueError(f"jacobian_mode invalid: {self.jacobian_mode!r}")
        if self.sampler not in ("none", "gauss", "roberts", "uniform"):
            raise ValueError(f"sampler invalid: {self.sampler!r}")
        if self.seed_source not in ("sim_joint_pos", "previous_solution", "default_pose"):
            raise ValueError(f"seed_source invalid: {self.seed_source!r}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")
        if self.sampler == "none" and self.n_seeds != 1:
            raise ValueError("sampler='none' requires n_seeds == 1")
