# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "MESH_APPROXIMATION_TOKENS",
    "activate_contact_sensors",
    "apply_articulation_root_properties",
    "apply_collision_properties",
    "apply_fixed_tendon_properties",
    "apply_mass_properties",
    "apply_drive",
    "apply_joint_drive_properties",
    "apply_mesh_collision",
    "apply_mesh_collision_properties",
    "apply_namespaced",
    "apply_rigid_body_properties",
    "apply_spatial_tendon_properties",
    "apply_surface_deformable_properties",
    "apply_volume_deformable_properties",
    "define_actuator_properties",
    "define_deformable_curve_properties",
    "CollisionFragment",
    "DeformableBodyFragment",
    "FixedTendonFragment",
    "MassCfg",
    "MassFragment",
    "JointDriveFragment",
    "ArticulationRootFragment",
    "MeshCollisionFragment",
    "OmniPhysicsDeformableBodyCfg",
    "RigidBodyFragment",
    "SchemaFragment",
    "SpatialTendonFragment",
    "UsdPhysicsCollisionCfg",
    "UsdPhysicsDriveCfg",
    "UsdPhysicsMeshCollisionCfg",
    "UsdPhysicsRigidBodyCfg",
    "NewtonMaterialPropertiesCfg",
]

from .schemas import (
    MESH_APPROXIMATION_TOKENS,
    activate_contact_sensors,
    apply_articulation_root_properties,
    apply_collision_properties,
    apply_fixed_tendon_properties,
    apply_mass_properties,
    apply_drive,
    apply_joint_drive_properties,
    apply_mesh_collision,
    apply_mesh_collision_properties,
    apply_namespaced,
    apply_rigid_body_properties,
    apply_spatial_tendon_properties,
    apply_surface_deformable_properties,
    apply_volume_deformable_properties,
    define_deformable_curve_properties,
)
from .schemas_actuators import (
    define_actuator_properties,
)
from .schemas_cfg import (
    ArticulationRootFragment,
    CollisionFragment,
    DeformableBodyFragment,
    FixedTendonFragment,
    MassCfg,
    MassFragment,
    JointDriveFragment,
    MeshCollisionFragment,
    OmniPhysicsDeformableBodyCfg,
    RigidBodyFragment,
    SchemaFragment,
    SpatialTendonFragment,
    UsdPhysicsCollisionCfg,
    UsdPhysicsDriveCfg,
    UsdPhysicsMeshCollisionCfg,
    UsdPhysicsRigidBodyCfg,
)

# Forwarded to isaaclab_newton.sim.schemas via __getattr__ shim
NewtonMaterialPropertiesCfg = ...
