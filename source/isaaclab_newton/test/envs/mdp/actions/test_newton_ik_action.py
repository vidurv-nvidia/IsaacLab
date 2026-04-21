# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Newton IK action term class wiring.

Runtime pipeline behavior (reset/step/no-NaN) is validated in the gym smoke
test under ``test/envs/`` (T12), which exercises the action term inside a
real ManagerBasedRLEnv. This file only verifies the class-level contract.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from isaaclab_newton.envs.mdp.actions import NewtonInverseKinematicsAction


def test_action_class_is_exported():
    """The action class must be importable and named correctly."""
    assert NewtonInverseKinematicsAction.__name__ == "NewtonInverseKinematicsAction"


def test_action_module_path():
    """The module containing the action class must exist."""
    import isaaclab_newton.envs.mdp.actions.newton_ik_actions as mod

    assert hasattr(mod, "NewtonInverseKinematicsAction")
