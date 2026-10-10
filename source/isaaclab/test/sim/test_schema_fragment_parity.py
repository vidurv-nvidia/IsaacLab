# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD authoring checks for composed schema fragments and their in-tree usage.

The tests run on an in-memory USD stage and intentionally do NOT launch Isaac Sim / Kit.
"""

import math

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.sim.schemas import UsdPhysicsDriveCfg, apply_joint_drive_properties

from isaaclab_physx.sim.schemas import PhysxJointCfg  # isort: skip

JOINT_PATH = "/World/Joint"


def _authored_attributes(prim: Usd.Prim) -> dict[str, object]:
    """Return every authored attribute on the prim, keyed by name."""
    return {attr.GetName(): attr.Get() for attr in prim.GetAuthoredAttributes()}


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
def test_joint_drive_fragments_author_axis_instance_and_units(joint_type):
    """The UsdPhysics drive + PhysX joint fragment pair authors the joint's drive instance in USD units.

    Running both joint types covers the two behaviours the joint-drive writer branches on: the
    ``DriveAPI:angular`` / ``DriveAPI:linear`` multi-instance selection, and the radian-to-degree
    conversion that applies to angular drives only.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    define = UsdPhysics.RevoluteJoint.Define if joint_type == "revolute" else UsdPhysics.PrismaticJoint.Define
    joint = define(stage, JOINT_PATH).GetPrim()

    apply_joint_drive_properties(
        JOINT_PATH,
        [
            UsdPhysicsDriveCfg(drive_type="force", max_force=87.0, stiffness=100.0, damping=10.0),
            PhysxJointCfg(max_joint_velocity=3.0),
        ],
        stage=stage,
    )

    instance = "angular" if joint_type == "revolute" else "linear"
    other_instance = "linear" if joint_type == "revolute" else "angular"
    authored = _authored_attributes(joint)
    assert f"drive:{instance}:physics:stiffness" in authored
    assert f"drive:{other_instance}:physics:stiffness" not in authored

    # angular drives are stored in degree units, linear drives in the cfg's own units
    scale = math.pi / 180.0 if joint_type == "revolute" else 1.0
    assert authored[f"drive:{instance}:physics:stiffness"] == pytest.approx(100.0 * scale)
    assert authored[f"drive:{instance}:physics:damping"] == pytest.approx(10.0 * scale)
    assert authored[f"drive:{instance}:physics:maxForce"] == pytest.approx(87.0)
    assert authored[f"drive:{instance}:physics:type"] == "force"
    velocity_scale = 180.0 / math.pi if joint_type == "revolute" else 1.0
    assert authored["physxJoint:maxJointVelocity"] == pytest.approx(3.0 * velocity_scale)


def test_asset_and_task_configs_use_bare_single_fragments(source_checkout_root):
    """Single schema fragments stay bare; lists are reserved for multiple fragments."""
    import ast

    slots = {
        "rigid_props",
        "collision_props",
        "articulation_props",
        "joint_drive_props",
        "mass_props",
        "mesh_collision_props",
        "fixed_tendons_props",
        "spatial_tendons_props",
        "physics_material",
    }
    violations = []
    for package in ("isaaclab_assets", "isaaclab_tasks"):
        for path in (source_checkout_root / "source" / package / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.keyword)
                    and node.arg in slots
                    and isinstance(node.value, ast.List)
                    and len(node.value.elts) == 1
                ):
                    violations.append(f"{path.relative_to(source_checkout_root)}:{node.lineno}: {node.arg}")
    assert not violations, "Unnecessary single-fragment lists:\n" + "\n".join(violations)
