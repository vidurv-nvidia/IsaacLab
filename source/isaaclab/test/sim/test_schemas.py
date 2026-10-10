# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import pytest
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg, PhysxTendonAxisRootCfg
from isaaclab_physx.sim.spawners.materials import (
    PhysxRigidBodyMaterialCfg,
    PhysxSurfaceDeformableBodyMaterialCfg,
)

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg, spawn_rigid_body_material

pytestmark = pytest.mark.integration


@pytest.fixture
def setup_simulation():
    """Fixture to set up and tear down the simulation context."""
    # Create a new stage
    sim_utils.create_new_stage()
    # Load kit helper
    sim = SimulationContext(SimulationCfg(dt=0.1))
    yield sim
    # Teardown
    sim._disable_app_control_on_stop_handle = True  # prevent timeout
    sim.stop()
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_rigid_body_material_base_cfg(setup_simulation):
    """Setting only UsdPhysics fields on RigidBodyMaterialBaseCfg must author the
    three friction/restitution attrs and must NOT apply PhysxMaterialAPI."""
    stage = sim_utils.get_current_stage()

    cfg = RigidBodyMaterialBaseCfg(static_friction=0.7, dynamic_friction=0.6, restitution=0.1)
    prim_path = "/World/Looks/BaseMaterial"
    spawn_rigid_body_material.__wrapped__(prim_path, cfg)

    prim = stage.GetPrimAtPath(prim_path)
    assert prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.7)
    assert prim.GetAttribute("physics:dynamicFriction").Get() == pytest.approx(0.6)
    assert prim.GetAttribute("physics:restitution").Get() == pytest.approx(0.1)
    applied = prim.GetAppliedSchemas()
    assert "PhysxMaterialAPI" not in applied, (
        f"PhysxMaterialAPI must not be applied for the base cfg; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_physx_rigid_body_material_cfg(setup_simulation):
    """Setting a PhysX-namespaced field on PhysxRigidBodyMaterialCfg must author the
    namespaced attribute AND apply PhysxMaterialAPI."""
    stage = sim_utils.get_current_stage()

    cfg = PhysxRigidBodyMaterialCfg(static_friction=0.7, compliant_contact_stiffness=100.0)
    prim_path = "/World/Looks/PhysxMaterial"
    spawn_rigid_body_material.__wrapped__(prim_path, cfg)

    prim = stage.GetPrimAtPath(prim_path)
    assert prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.7)
    assert prim.GetAttribute("physxMaterial:compliantContactStiffness").Get() == pytest.approx(100.0)
    applied = prim.GetAppliedSchemas()
    assert "PhysxMaterialAPI" in applied, (
        f"PhysxMaterialAPI must be applied when a PhysX field is set; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_deformable_collision_props_land_on_simulation_mesh(setup_simulation):
    """Regression: ``collision_props`` on a deformable spawner must author ``physxCollision:*``
    on the simulation mesh, which is the prim carrying ``UsdPhysics.CollisionAPI``. Authoring
    them on the deformable body prim leaves them inert."""
    stage = sim_utils.get_current_stage()

    cfg = sim_utils.MeshCuboidCfg(
        size=(0.3, 0.04, 0.04),
        # the surface slot needs no tetrahedralization dependency
        surface_deformable_props=[],
        collision_props=PhysxCollisionCfg(contact_offset=0.005, rest_offset=0.0005),
        physics_material=PhysxSurfaceDeformableBodyMaterialCfg(),
    )
    cfg.func("/World/beam_dc", cfg)

    sim_mesh_prim = stage.GetPrimAtPath("/World/beam_dc/sim_mesh")
    assert "PhysxCollisionAPI" in sim_mesh_prim.GetAppliedSchemas()
    assert sim_mesh_prim.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.005)
    assert sim_mesh_prim.GetAttribute("physxCollision:restOffset").Get() == pytest.approx(0.0005)
    body_prim = stage.GetPrimAtPath("/World/beam_dc")
    assert not body_prim.GetAttribute("physxCollision:restOffset").HasAuthoredValue()


@pytest.mark.isaacsim_ci
def test_activate_contact_sensors_nested_rigid_bodies(setup_simulation):
    """Test contact-report schemas are applied to nested rigid-body trees."""
    stage = sim_utils.get_current_stage()

    rigid_body_paths = [
        "/World/Robot/Geometry/pelvis",
        "/World/Robot/Geometry/pelvis/left_hip",
        "/World/Robot/Geometry/pelvis/left_hip/left_knee",
    ]
    sim_utils.create_prim("/World/Robot", prim_type="Xform")
    sim_utils.create_prim("/World/Robot/Geometry", prim_type="Xform")
    for prim_path in rigid_body_paths:
        sim_utils.create_prim(prim_path, prim_type="Xform")
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(prim_path))

    schemas.activate_contact_sensors("/World/Robot", threshold=2.5)

    for prim_path in rigid_body_paths:
        prim = stage.GetPrimAtPath(prim_path)
        applied_schemas = prim.GetAppliedSchemas()
        assert "PhysxRigidBodyAPI" in applied_schemas
        assert "PhysxContactReportAPI" in applied_schemas
        assert prim.GetAttribute("physxRigidBody:sleepThreshold").Get() == pytest.approx(0.0)
        assert prim.GetAttribute("physxContactReport:threshold").Get() == pytest.approx(2.5)


@pytest.mark.isaacsim_ci
def test_rigid_body_and_mass_fragments_nested_rigid_bodies(setup_simulation):
    """Test rigid-body and mass fragments are applied to nested rigid-body trees."""
    stage = sim_utils.get_current_stage()

    rigid_body_paths = [
        "/World/Robot/Geometry/pelvis",
        "/World/Robot/Geometry/pelvis/left_hip",
        "/World/Robot/Geometry/pelvis/left_hip/left_knee",
    ]
    sim_utils.create_prim("/World/Robot", prim_type="Xform")
    sim_utils.create_prim("/World/Robot/Geometry", prim_type="Xform")
    for prim_path in rigid_body_paths:
        sim_utils.create_prim(prim_path, prim_type="Xform")
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(prim_path))
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(prim_path))

    assert schemas.apply_rigid_body_properties("/World/Robot(/.*)?", [PhysxRigidBodyCfg(disable_gravity=True)])
    assert schemas.apply_mass_properties("/World/Robot(/.*)?", [schemas.MassCfg(mass=2.5)])

    for prim_path in rigid_body_paths:
        prim = stage.GetPrimAtPath(prim_path)
        assert prim.GetAttribute("physxRigidBody:disableGravity").Get() is True, f"Failed for {prim_path}"
        assert prim.GetAttribute("physics:mass").Get() == pytest.approx(2.5), f"Failed for {prim_path}"


@pytest.mark.isaacsim_ci
def test_multi_instance_schema_detection_on_tendon_joints(setup_simulation):
    """Test that multi-instance PhysX tendon schema tokens are recognized with their instance suffixes.

    Multi-instance schemas (e.g. PhysxTendonAxisAPI, PhysxTendonAxisRootAPI) appear in
    GetAppliedSchemas() as 'SchemaName:instanceName' (e.g. 'PhysxTendonAxisAPI:inst0').
    An exact ``in list`` check fails because 'PhysxTendonAxisAPI' != 'PhysxTendonAxisAPI:inst0'.
    This test ensures both the joint-drive skip predicate and the fixed-tendon writer handle
    multiple-apply schema tokens correctly.
    """
    stage = sim_utils.get_current_stage()
    drive_fragments = [schemas.UsdPhysicsDriveCfg(drive_type="acceleration", stiffness=10.0, damping=0.1)]

    # -- set up two body prims connected by a revolute joint
    sim_utils.create_prim("/World/tendon_test", prim_type="Xform")
    sim_utils.create_prim("/World/tendon_test/body0", prim_type="Cube")
    sim_utils.create_prim("/World/tendon_test/body1", prim_type="Cube")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/tendon_test/body1/joint0")
    joint_prim = joint.GetPrim()
    joint_path = joint_prim.GetPrimPath().pathString

    # -- 1) Joint with only tendon child schema (no root) -> drive should be SKIPPED
    joint_prim.AddAppliedSchema("PhysxTendonAxisAPI:inst0")
    applied = joint_prim.GetAppliedSchemas()
    assert any("PhysxTendonAxisAPI" in s for s in applied), "Multi-instance schema not found via substring"
    assert "PhysxTendonAxisAPI" not in applied, "Exact match should NOT find multi-instance schema"
    assert not schemas.apply_joint_drive_properties(joint_path, drive_fragments, stage)
    assert not UsdPhysics.DriveAPI(joint_prim, "angular"), "Tendon child joint should be skipped"

    # -- 2) Joint with both child AND root tendon schema -> drive should NOT be skipped
    joint_prim.AddAppliedSchema("PhysxTendonAxisRootAPI:inst0")
    applied = joint_prim.GetAppliedSchemas()
    assert any("PhysxTendonAxisRootAPI" in s for s in applied)
    assert "PhysxTendonAxisRootAPI" not in applied, "Exact match should NOT find multi-instance schema"
    assert schemas.apply_joint_drive_properties(joint_path, drive_fragments, stage), "Tendon root should be driven"

    # -- 3) the fixed-tendon writer detects the multi-instance root schema
    tendon_fragments = [PhysxTendonAxisRootCfg(stiffness=10.0, damping=0.1)]
    assert schemas.apply_fixed_tendon_properties(joint_path, tendon_fragments, stage)
    assert joint_prim.GetAttribute("physxTendon:inst0:stiffness").Get() == pytest.approx(10.0)

    # -- 4) Prim WITHOUT any tendon root schema -> nothing to tune
    sim_utils.create_prim("/World/tendon_test/body2", prim_type="Cube")
    no_tendon_joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/tendon_test/body2/joint1")
    assert not schemas.apply_fixed_tendon_properties(
        no_tendon_joint.GetPrim().GetPrimPath().pathString, tendon_fragments, stage
    )


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("kind", ["volume", "surface"])
def test_physx_deformable_fragments(setup_simulation, kind):
    """Create OmniPhysics rest/bind poses and compose PhysX body and material fragments."""
    from isaaclab_physx.physics import PhysxManager
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyCfg, PhysxSurfaceDeformableBodyCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableMaterialCfg, PhysxSurfaceDeformableMaterialCfg

    from pxr import Usd, UsdGeom

    from isaaclab.sim.spawners.materials import spawn_physics_material_from_fragments

    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/Body").GetPrim()
    visual = UsdGeom.Mesh.Define(stage, "/Body/visual")
    visual.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh = UsdGeom.TetMesh.Define(stage, "/Body/sim") if kind == "volume" else UsdGeom.Mesh.Define(stage, "/Body/sim")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    if kind == "volume":
        mesh.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
    else:
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        mesh.CreateFaceVertexCountsAttr([3])
    PhysxManager.setup_deformable_body(body, kind, mesh.GetPrim(), visual.GetPrim())
    assert body.HasAPI("OmniPhysicsDeformableBodyAPI")
    assert mesh.GetPrim().HasAPI(f"OmniPhysics{kind.title()}DeformableSimAPI")
    assert mesh.GetPrim().GetAttribute("omniphysics:restShapePoints").Get() == mesh.GetPointsAttr().Get()
    rest_indices = "restTetVtxIndices" if kind == "volume" else "restTriVtxIndices"
    assert len(mesh.GetPrim().GetAttribute("omniphysics:" + rest_indices).Get()) == 1
    assert (
        visual.GetPrim().GetAttribute("deformablePose:default:omniphysics:points").Get() == visual.GetPointsAttr().Get()
    )
    body_cfg = PhysxDeformableBodyCfg(solver_position_iteration_count=32, linear_damping=0.1)
    assert body_cfg.func(body_cfg, "/Body", stage)
    assert body.GetAttribute("physxDeformableBody:solverPositionIterationCount").Get() == 32
    assert body.GetAttribute("physxDeformableBody:linearDamping").Get() == pytest.approx(0.1)
    materials = [PhysxDeformableMaterialCfg(elasticity_damping=0.005)]
    if kind == "surface":
        surface_cfg = PhysxSurfaceDeformableBodyCfg(collision_pair_update_frequency=2)
        assert surface_cfg.func(surface_cfg, "/Body", stage)
        assert body.GetAttribute("physxDeformableBody:collisionPairUpdateFrequency").Get() == 2
        materials.append(PhysxSurfaceDeformableMaterialCfg(bend_damping=0.1))
    material = spawn_physics_material_from_fragments("/Material", materials, stage)
    assert material.GetAttribute("physxDeformableMaterial:elasticityDamping").Get() == pytest.approx(0.005)
    if kind == "surface":
        assert material.GetAttribute("physxDeformableMaterial:bendDamping").Get() == pytest.approx(0.1)
