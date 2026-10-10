* Removed the Newton single schema cfgs deprecated in 3.1. Compose the schema fragments in the
  spawner slot instead:

  * ``NewtonRigidBodyPropertiesCfg`` → :class:`~isaaclab.sim.schemas.UsdPhysicsRigidBodyCfg` plus
    :class:`~isaaclab_physx.sim.schemas.PhysxRigidBodyCfg` for ``disable_gravity``;
    ``MujocoRigidBodyPropertiesCfg`` → the same plus
    :class:`~isaaclab_newton.sim.schemas.MujocoRigidBodyCfg` for ``gravcomp``.
  * ``NewtonJointDrivePropertiesCfg`` → :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` plus
    :class:`~isaaclab_physx.sim.schemas.PhysxJointCfg` for ``max_joint_velocity``;
    ``MujocoJointDrivePropertiesCfg`` → the same plus
    :class:`~isaaclab_newton.sim.schemas.MujocoJointCfg` for ``actuatorgravcomp``.
  * ``NewtonCollisionPropertiesCfg`` → :class:`~isaaclab.sim.schemas.UsdPhysicsCollisionCfg`,
    :class:`~isaaclab_physx.sim.schemas.PhysxCollisionCfg` and
    :class:`~isaaclab_newton.sim.schemas.NewtonCollisionCfg` in ``collision_props``.
  * ``NewtonMeshCollisionPropertiesCfg`` → the collision fragments above plus
    :class:`~isaaclab.sim.schemas.UsdPhysicsMeshCollisionCfg` and
    :class:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionCfg` in ``mesh_collision_props``.
  * ``NewtonSDFCollisionPropertiesCfg`` → the collision fragments above plus
    :class:`~isaaclab_newton.sim.schemas.NewtonSDFCollisionCfg` in ``mesh_collision_props``.
  * ``NewtonArticulationRootPropertiesCfg`` →
    :class:`~isaaclab_physx.sim.schemas.PhysxArticulationCfg` (for ``articulation_enabled``) plus
    :class:`~isaaclab_newton.sim.schemas.NewtonArticulationCfg`.
  * ``NewtonDeformableBodyPropertiesCfg`` → an empty ``volume_deformable_props`` or
    ``surface_deformable_props`` slot, which creates the body with Newton's defaults.
