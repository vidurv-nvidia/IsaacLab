# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the backend split of deformable materials."""

import dataclasses

import pytest
from isaaclab_newton.sim.spawners.materials import (
    NewtonDeformableBodyMaterialCfg,
    NewtonDeformableMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_physx.sim.spawners.materials import (
    PhysxDeformableBodyMaterialCfg,
    PhysXDeformableMaterialCfg,
    PhysxSurfaceDeformableBodyMaterialCfg,
)

import isaaclab.sim.spawners.materials.physics_materials_cfg as core_materials_cfg
from isaaclab.sim.spawners.materials import (
    DeformableBodyMaterialBaseCfg,
    SurfaceDeformableBodyMaterialBaseCfg,
)

pytestmark = pytest.mark.integration


def _field_names(cls) -> set[str]:
    return {field.name for field in dataclasses.fields(cls)}


def _assert_no_property_prefix_field(cls):
    assert "_property_prefix" not in _field_names(cls)


def test_common_deformable_material_cfg_has_no_backend_fields():
    """Common deformable material bases are empty backend extension points."""
    fields = _field_names(DeformableBodyMaterialBaseCfg)
    surface_fields = _field_names(SurfaceDeformableBodyMaterialBaseCfg)

    assert fields == {"func"}
    assert surface_fields == {"func"}
    assert not hasattr(core_materials_cfg, "PhysXDeformableMaterialCfg")
    assert not hasattr(core_materials_cfg, "NewtonDeformableMaterialCfg")
    assert not hasattr(DeformableBodyMaterialBaseCfg, "_usd_namespace")
    assert not hasattr(DeformableBodyMaterialBaseCfg, "_usd_applied_schema")
    assert not hasattr(SurfaceDeformableBodyMaterialBaseCfg, "_usd_namespace")
    assert not hasattr(SurfaceDeformableBodyMaterialBaseCfg, "_usd_applied_schema")
    _assert_no_property_prefix_field(DeformableBodyMaterialBaseCfg)
    _assert_no_property_prefix_field(SurfaceDeformableBodyMaterialBaseCfg)


def test_physx_deformable_material_cfgs_use_core_material_function():
    """PhysX deformable materials own PhysX fields while the material function stays in core."""
    material = PhysxDeformableBodyMaterialCfg()
    surface_material = PhysxSurfaceDeformableBodyMaterialCfg()

    assert str(material.func) == "isaaclab.sim.spawners.materials.physics_materials:spawn_deformable_body_material"
    assert str(surface_material.func) == str(material.func)
    _assert_no_property_prefix_field(type(material))
    _assert_no_property_prefix_field(type(surface_material))
    assert {"density", "static_friction", "dynamic_friction", "youngs_modulus", "poissons_ratio"}.issubset(
        _field_names(type(material))
    )
    assert "surface_thickness" in _field_names(type(surface_material))
    assert "bend_damping" in _field_names(type(surface_material))
    assert PhysXDeformableMaterialCfg._usd_namespace == "physxDeformableMaterial"
    assert PhysXDeformableMaterialCfg._usd_applied_schema == "PhysxDeformableMaterialAPI"
    assert "_usd_applied_schema" not in type(material).__dict__
    assert type(surface_material)._usd_namespace == "physxDeformableMaterial"
    assert type(surface_material)._usd_applied_schema == "PhysxSurfaceDeformableMaterialAPI"


def test_newton_deformable_material_cfgs_use_core_material_function():
    """Newton deformable materials own Newton fields while the material function stays in core."""
    material = NewtonDeformableBodyMaterialCfg()
    surface_material = NewtonSurfaceDeformableBodyMaterialCfg()

    assert str(material.func) == "isaaclab.sim.spawners.materials.physics_materials:spawn_deformable_body_material"
    assert str(surface_material.func) == str(material.func)
    _assert_no_property_prefix_field(type(material))
    _assert_no_property_prefix_field(type(surface_material))
    assert "youngs_modulus" not in _field_names(type(material))
    assert "poissons_ratio" not in _field_names(type(material))
    assert {"density", "particle_radius", "k_mu", "k_lambda", "k_damp"}.issubset(_field_names(type(material)))
    assert NewtonDeformableMaterialCfg._usd_namespace == "newton"
    assert NewtonDeformableMaterialCfg._usd_applied_schema is None
