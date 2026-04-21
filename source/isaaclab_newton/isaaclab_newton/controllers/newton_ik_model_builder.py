# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Builder for a single-arm Newton :class:`~newton.Model` used by Newton IK.

The Newton IK solver expects each of its ``n_problems`` to be a replication of
a single articulation topology. The IsaacLab simulation Newton model instead
holds N articulations concatenated, so we build a small dedicated IK model
from the same USD asset that the sim uses. Topology must match the per-env
arm slice of the sim model; we validate this at construction time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import newton
import warp as wp
from newton import ModelBuilder
from pxr import Usd

from isaaclab.assets import ArticulationCfg


@dataclass
class IKModelInfo:
    """Metadata for a single-arm IK Newton model bound to an IsaacLab asset.

    Attributes:
        ee_link_index: Index into ``ik_model.body_q`` for the end-effector body.
        arm_dof_count: Number of arm joint DOFs controlled by IK.
        ik_joint_names: Names of the IK-controlled joints, in IK-model order.
        sim_to_ik_joint_perm: Permutation mapping IsaacLab sim arm joint
            indices to IK-model joint indices, length ``arm_dof_count``.
    """

    ee_link_index: int
    arm_dof_count: int
    ik_joint_names: list[str]
    sim_to_ik_joint_perm: list[int]


def _label_short_name(label: str) -> str:
    """Return the final path component of a USD-path label.

    Newton body/joint labels may be full USD paths such as
    ``'/panda/panda_link0/panda_joint1'``.  This helper strips the prefix so
    that caller-facing names stay short (e.g. ``'panda_joint1'``).
    """
    return label.rsplit("/", 1)[-1]


def _resolve_joint_names(all_labels: list[str], patterns: list[str]) -> list[str]:
    """Resolve IsaacLab-style regex patterns against a list of joint labels.

    Matching is performed against the short name (last path component) of each
    label so callers can write plain names like ``panda_joint.*`` even when the
    Newton model stores full USD paths.
    """
    resolved: list[str] = []
    for label in all_labels:
        short = _label_short_name(label)
        if any(re.fullmatch(pat, short) for pat in patterns):
            resolved.append(label)
    return resolved


def build_single_arm_ik_model(
    asset_cfg: ArticulationCfg,
    body_name: str,
    joint_names: list[str],
    device: str,
) -> tuple[newton.Model, IKModelInfo]:
    """Build a Newton :class:`~newton.Model` containing one copy of the robot.

    Loads the same USD asset used by the sim into a fresh
    :class:`~newton.ModelBuilder`, finalizes a single-articulation
    :class:`~newton.Model` with the base at the world origin, and returns
    metadata needed by :class:`~isaaclab_newton.controllers.NewtonIKController`.

    Newton body and joint labels may be full USD paths (e.g.
    ``'/panda/panda_hand'``).  The ``body_name`` and ``joint_names`` patterns
    are matched against the short name (last path component), so callers should
    use simple names such as ``'panda_hand'`` and ``'panda_joint.*'``.

    For instanceable USDs the Newton builder may produce a small extra
    articulation for the world-root fixed joint.  The function automatically
    selects the largest articulation (by joint count) as the arm.

    Args:
        asset_cfg: IsaacLab articulation configuration whose ``spawn.usd_path``
            is loaded.
        body_name: Short name of the end-effector body in the asset (last USD
            path component, e.g. ``'panda_hand'``).
        joint_names: Joint name regex patterns controlled by IK, matched
            against short names (e.g. ``['panda_joint.*']``).
        device: Warp device string (e.g. ``"cuda:0"`` or ``"cpu"``).

    Returns:
        Tuple of the finalized single-articulation :class:`~newton.Model` and
        the :class:`IKModelInfo` describing the EE body and joint mapping.

    Raises:
        ValueError: If the EE body or any IK-controlled joints cannot be
            resolved from the loaded model.
        ValueError: If the loaded USD does not produce any articulations.
    """
    usd_path = asset_cfg.spawn.usd_path
    builder = ModelBuilder()
    stage = Usd.Stage.Open(usd_path)
    # Force fixed base at world origin so world-frame targets equal base-frame targets.
    builder.add_usd(
        stage,
        floating=False,
        xform=wp.transform_identity(),
        override_root_xform=True,
    )

    model = builder.finalize(device=device, requires_grad=True)

    if model.articulation_count < 1:
        raise ValueError(
            f"Expected USD '{usd_path}' to produce at least 1 articulation, "
            f"got {model.articulation_count}."
        )

    # Resolve body names (matching by short name / last USD path component).
    body_labels = list(model.body_label)
    body_short_names = [_label_short_name(lbl) for lbl in body_labels]
    if body_name not in body_short_names:
        raise ValueError(
            f"Body name '{body_name}' not found in IK model. "
            f"Available (short names): {body_short_names}"
        )
    ee_link_index = body_short_names.index(body_name)

    # Resolve joint names (matching by short name / last USD path component).
    all_joint_labels = list(model.joint_label)
    ik_joint_labels = _resolve_joint_names(all_joint_labels, joint_names)
    if not ik_joint_labels:
        all_short = [_label_short_name(lbl) for lbl in all_joint_labels]
        raise ValueError(
            f"No joints in IK model match {joint_names}. "
            f"Available (short names): {all_short}"
        )
    ik_joint_names = [_label_short_name(lbl) for lbl in ik_joint_labels]
    arm_dof_count = len(ik_joint_names)
    sim_to_ik_joint_perm = [all_joint_labels.index(lbl) for lbl in ik_joint_labels]

    info = IKModelInfo(
        ee_link_index=ee_link_index,
        arm_dof_count=arm_dof_count,
        ik_joint_names=ik_joint_names,
        sim_to_ik_joint_perm=sim_to_ik_joint_perm,
    )
    return model, info
