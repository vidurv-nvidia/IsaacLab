# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration checks for the heterogeneous-motion customer demonstrator."""

import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import pytest

_PACKAGE_DIR = Path(__file__).parents[2] / "isaaclab_tasks" / "contrib" / "heterogeneous_motion"


def _register(module_name: str, module_path: Path) -> None:
    """Execute one registration module without importing simulator-dependent task code."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


_register("heterogeneous_motion_registration", _PACKAGE_DIR / "__init__.py")
_register("heterogeneous_motion_examples_registration", _PACKAGE_DIR / "examples" / "__init__.py")


@pytest.mark.parametrize(
    "task_id",
    (
        "IsaacContrib-Heterogeneous-Maze-Motion-Direct",
        "IsaacContrib-Heterogeneous-Motion-Direct",
        "IsaacContrib-Heterogeneous-Motion",
        "IsaacContrib-Heterogeneous-FixedSlots-MultiUsd-Direct",
        "IsaacContrib-Heterogeneous-Unequal-Compound-Direct",
        "IsaacContrib-Heterogeneous-Unequal-InclusionSet-Direct",
        "IsaacContrib-Heterogeneous-Unequal-SceneAdd-Direct",
    ),
)
def test_heterogeneous_motion_registers_documented_examples(task_id: str):
    """Register every documented heterogeneous construction example."""
    spec = gym.spec(task_id)
    assert spec.kwargs["env_cfg_entry_point"]
