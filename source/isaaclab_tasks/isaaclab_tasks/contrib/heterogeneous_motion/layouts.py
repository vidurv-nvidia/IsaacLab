# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-owned associations between declared maze layouts and motion files."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.cloner import InclusionSet

from .motion import LOWER_MOTION_FILE, UPPER_MOTION_FILE

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene


@dataclass(frozen=True)
class LayoutSpec:
    """Physical inclusion set paired with task-owned motion data."""

    inclusion: InclusionSet
    """Physical assets active in this exact maze layout."""

    motion_file: Path
    """Planar motion-reference CSV for this physical layout."""


_COMMON_ASSETS = ("agent", "top_wall", "bottom_wall", "left_wall", "right_wall")

LAYOUTS = (
    LayoutSpec(
        inclusion=InclusionSet(assets=[*_COMMON_ASSETS, "upper_divider"]),
        motion_file=UPPER_MOTION_FILE,
    ),
    LayoutSpec(
        inclusion=InclusionSet(assets=[*_COMMON_ASSETS, "lower_divider_lower", "lower_divider_upper"]),
        motion_file=LOWER_MOTION_FILE,
    ),
)
"""Declared maze layouts and their task-owned motion references."""


def resolve_layout_ids(
    scene: InteractiveScene, layouts: Sequence[LayoutSpec] = LAYOUTS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive compact physical groups and declared layout IDs from a scene's clone mask.

    Compact combination IDs only express equality between physical signatures. The
    returned layout IDs follow ``layouts`` order and can safely index task-owned data.

    Args:
        scene: Constructed heterogeneous scene.
        layouts: Declared physical layouts paired with task data.

    Returns:
        A tuple containing unique clone-mask rows, per-environment compact combination
        IDs, and per-environment declared layout IDs.
    """
    plan = scene.clone_plan
    if plan is None or plan.clone_mask.shape[0] == 0:
        raise RuntimeError("The scene did not produce a clone mask.")

    combination_rows, combination_ids = torch.unique(plan.clone_mask.T, dim=0, return_inverse=True)
    asset_names = tuple(dict.fromkeys(asset_name for layout in layouts for asset_name in layout.inclusion.assets))
    if not asset_names:
        raise ValueError("Expected at least one declared layout asset.")

    declared_signatures: dict[frozenset[str], int] = {}
    for layout_index, layout in enumerate(layouts):
        signature = frozenset(layout.inclusion.assets)
        if signature in declared_signatures:
            raise ValueError("Each LayoutSpec must declare a unique physical asset signature.")
        declared_signatures[signature] = layout_index

    presence_columns = []
    for asset_name in asset_names:
        asset_cfg = getattr(scene.cfg, asset_name, None)
        rows = () if asset_cfg is None else plan.cfg_rows.get(id(asset_cfg), ())
        if not rows:
            raise ValueError(f"Layout asset {asset_name!r} is missing from clone_plan.cfg_rows.")
        presence_columns.append(combination_rows[:, list(rows)].any(dim=1))

    active_by_combination = torch.stack(presence_columns, dim=1).cpu().tolist()
    combination_to_layout = []
    for active_assets in active_by_combination:
        signature = frozenset(asset_name for asset_name, is_active in zip(asset_names, active_assets) if is_active)
        layout_id = declared_signatures.get(signature)
        if layout_id is None:
            raise ValueError(f"Observed undeclared physical asset signature: {sorted(signature)}.")
        combination_to_layout.append(layout_id)

    lookup = torch.tensor(combination_to_layout, dtype=torch.long, device=combination_ids.device)
    return combination_rows, combination_ids, lookup[combination_ids]
