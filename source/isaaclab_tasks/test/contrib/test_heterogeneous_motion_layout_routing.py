# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task-local heterogeneous layout routing."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab.cloner import InclusionSet

from isaaclab_tasks.contrib.heterogeneous_motion.layouts import LayoutSpec, resolve_layout_ids


def _make_scene(clone_mask: torch.Tensor) -> SimpleNamespace:
    """Build a minimal scene carrying two named clone-plan assets."""
    first_cfg = object()
    second_cfg = object()
    cfg = SimpleNamespace(first=first_cfg, second=second_cfg)
    plan = SimpleNamespace(
        clone_mask=clone_mask,
        cfg_rows={id(first_cfg): (0,), id(second_cfg): (1,)},
    )
    return SimpleNamespace(cfg=cfg, clone_plan=plan)


def _layouts() -> tuple[LayoutSpec, LayoutSpec]:
    """Return two semantic layouts in an order independent of compact IDs."""
    return (
        LayoutSpec(InclusionSet(assets=["first"]), Path("first.csv")),
        LayoutSpec(InclusionSet(assets=["second"]), Path("second.csv")),
    )


def test_resolve_layout_ids_preserves_declared_semantics() -> None:
    """Map arbitrary compact IDs back to declared layout order."""
    clone_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, True]],
        dtype=torch.bool,
    )

    rows, combination_ids, layout_ids = resolve_layout_ids(_make_scene(clone_mask), _layouts())

    assert torch.equal(rows[combination_ids], clone_mask.T)
    assert combination_ids[0] == combination_ids[2]
    assert combination_ids[1] == combination_ids[3]
    assert combination_ids[0] != combination_ids[1]
    assert layout_ids.tolist() == [0, 1, 0, 1]


def test_resolve_layout_ids_rejects_duplicate_signatures() -> None:
    """Reject ambiguous task data for the same physical signature."""
    clone_mask = torch.tensor([[True], [False]], dtype=torch.bool)
    layouts = (
        LayoutSpec(InclusionSet(assets=["first"]), Path("first.csv")),
        LayoutSpec(InclusionSet(assets=["first"]), Path("duplicate.csv")),
    )

    with pytest.raises(ValueError, match="unique physical asset signature"):
        resolve_layout_ids(_make_scene(clone_mask), layouts)


def test_resolve_layout_ids_rejects_undeclared_signature() -> None:
    """Reject an observed physical combination without task data."""
    clone_mask = torch.tensor([[True], [True]], dtype=torch.bool)

    with pytest.raises(ValueError, match="Observed undeclared physical asset signature"):
        resolve_layout_ids(_make_scene(clone_mask), _layouts())
