# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime validation entry point for the heterogeneous maze demonstrator.

Run with ``./isaaclab.sh -p -m isaaclab_tasks.contrib.heterogeneous_motion.validation``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from isaaclab.app import AppLauncher

TASK_ID = "IsaacContrib-Heterogeneous-Maze-Motion-Direct"


def _assert_finite(tensors: Mapping[str, torch.Tensor] | torch.Tensor) -> None:
    """Assert that every tensor in an observation mapping is finite."""
    values = tensors.values() if isinstance(tensors, Mapping) else (tensors,)
    for value in values:
        if not torch.isfinite(value).all():
            raise AssertionError("Encountered non-finite environment output.")


def _assert_layout_routing(env: Any) -> dict[str, int]:
    """Validate physical grouping, semantic motion routing, and completion."""
    plan = env.scene.clone_plan
    if plan is None:
        raise AssertionError("The scene did not publish a clone plan.")

    physical_rows = plan.clone_mask.T
    if not torch.equal(env.combination_rows[env.combination_ids], physical_rows):
        raise AssertionError("Compact combination IDs do not reconstruct the physical clone signatures.")

    derived_rows, derived_ids = torch.unique(physical_rows, dim=0, return_inverse=True)
    if not torch.equal(derived_rows, env.combination_rows) or not torch.equal(derived_ids, env.combination_ids):
        raise AssertionError("The task grouping does not match clone_mask.T.")

    even_ids = env.combination_ids[0::2]
    odd_ids = env.combination_ids[1::2]
    if (
        env.combination_ids[0] == env.combination_ids[1]
        or not torch.all(even_ids == even_ids[0])
        or not torch.all(odd_ids == odd_ids[0])
    ):
        raise AssertionError("Expected alternating, distinct physical signature groups.")

    expected_layout_ids = torch.arange(env.num_envs, device=env.device) % len(env.motion_files)
    if not torch.equal(env.layout_ids, expected_layout_ids):
        raise AssertionError("Physical signatures do not map to the declared layout order.")

    counts = torch.bincount(env.layout_ids, minlength=len(env.motion_files))
    if counts.numel() != 2 or int(counts.min()) != env.num_envs // 2:
        raise AssertionError("Expected two round-robin physical layout groups.")
    if int(counts.max()) - int(counts.min()) > 1:
        raise AssertionError("Physical layout groups are not round-robin balanced.")

    expected_files = ("maze_upper.csv", "maze_lower.csv")
    if tuple(Path(path).name for path in env.motion_files) != expected_files:
        raise AssertionError("The layout table does not match the declared motion files.")

    goals = env._goal_positions()
    reference_velocity, path_distance_squared = env._reference_velocity(goals)
    if not torch.allclose(reference_velocity, torch.zeros_like(reference_velocity)):
        raise AssertionError("Reference controller does not complete at the final motion sample.")
    if not torch.allclose(path_distance_squared, torch.zeros_like(path_distance_squared)):
        raise AssertionError("Final motion sample is not reachable through the padded path bank.")
    return {Path(env.motion_files[index]).stem: int(count) for index, count in enumerate(counts)}


def validate_maze(env: Any, steps: int) -> dict[str, int]:
    """Run zero/random-action, reset-stability, and layout-routing checks.

    Args:
        env: Constructed heterogeneous maze environment.
        steps: Number of random-action steps after the zero-action smoke.

    Returns:
        Per-motion-file physical layout counts.
    """
    task_env = env.unwrapped
    num_envs = task_env.num_envs
    observations, _ = env.reset()
    if observations["policy"].shape != (num_envs, 6):
        raise AssertionError(f"Expected policy observations [num_envs, 6], got {observations['policy'].shape}.")
    _assert_finite(observations)
    initial_combination_ids = task_env.combination_ids.clone()
    initial_layout_ids = task_env.layout_ids.clone()
    groups = _assert_layout_routing(task_env)

    zero_actions = torch.zeros((num_envs, 2), device=task_env.device)
    observations, rewards, terminated, truncated, _ = env.step(zero_actions)
    _assert_finite(observations)
    _assert_finite(rewards)
    if rewards.shape != (num_envs,):
        raise AssertionError(f"Expected rewards [num_envs], got {rewards.shape}.")
    if terminated.shape != (num_envs,) or truncated.shape != (num_envs,):
        raise AssertionError("Termination signals have an unexpected shape.")

    for _ in range(steps):
        random_actions = 2.0 * torch.rand_like(zero_actions) - 1.0
        observations, rewards, _, _, _ = env.step(random_actions)
        _assert_finite(observations)
        _assert_finite(rewards)

    env.reset()
    if not torch.equal(task_env.combination_ids, initial_combination_ids):
        raise AssertionError("Physical combination IDs changed across reset.")
    if not torch.equal(task_env.layout_ids, initial_layout_ids):
        raise AssertionError("Layout-to-motion routing changed across reset.")
    return groups


def main() -> None:
    """Construct, validate, and close the maze environment."""
    parser = argparse.ArgumentParser(description="Validate the heterogeneous maze demonstrator.")
    parser.add_argument("--num-envs", type=int, default=2, choices=(2, 4, 1000))
    parser.add_argument("--steps", type=int, default=100)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    simulation_app = AppLauncher(args_cli).app

    import gymnasium as gym  # noqa: PLC0415

    import isaaclab_tasks.contrib.heterogeneous_motion  # noqa: F401, PLC0415
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: PLC0415

    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        groups = validate_maze(env, args_cli.steps)
        print(f"Validated {args_cli.num_envs} environments: {groups}")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
