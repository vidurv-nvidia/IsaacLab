# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Newton inverse-kinematics action term."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

from isaaclab_newton.controllers import NewtonIKControllerCfg


@configclass
class NewtonInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for the Newton IK action term."""

    @configclass
    class OffsetCfg:
        """Pose offset from the parent body frame to the constrained EE frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent body [m]."""

        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion ``(x, y, z, w)`` w.r.t. the parent body."""

    class_type: type | str = "{DIR}.newton_ik_actions:NewtonInverseKinematicsAction"

    joint_names: list[str] = MISSING
    """Joint name regex patterns controlled by IK."""

    body_name: str = MISSING
    """Name of the EE body for which IK is performed."""

    body_offset: OffsetCfg | None = None
    """Optional pose offset from the EE body frame to the constrained frame."""

    scale: float | tuple[float, ...] = 1.0
    """Scale factor applied to the input action."""

    clip: dict[str, tuple[float, float]] | None = None
    """Optional per-joint clip dict."""

    controller: NewtonIKControllerCfg = MISSING
    """Configuration for the underlying :class:`NewtonIKController`."""
