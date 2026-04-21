# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the isaaclab_newton.controllers package surface."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import isaaclab_newton.controllers as controllers


def test_package_imports():
    assert controllers is not None
