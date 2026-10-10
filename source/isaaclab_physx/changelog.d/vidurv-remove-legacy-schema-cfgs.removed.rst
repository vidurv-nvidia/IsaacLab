* Removed the PhysX single schema cfgs deprecated in 3.1. Compose the schema fragments in the
  spawner slot instead:

  * ``PhysxRigidBodyPropertiesCfg`` and ``RigidBodyPropertiesCfg`` →
    :class:`~isaaclab.sim.schemas.UsdPhysicsRigidBodyCfg` plus
    :class:`~isaaclab_physx.sim.schemas.PhysxRigidBodyCfg`.
  * ``PhysxCollisionPropertiesCfg`` and ``CollisionPropertiesCfg`` →
    :class:`~isaaclab.sim.schemas.UsdPhysicsCollisionCfg` plus
    :class:`~isaaclab_physx.sim.schemas.PhysxCollisionCfg`.
  * ``PhysxJointDrivePropertiesCfg`` and ``JointDrivePropertiesCfg`` →
    :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` plus
    :class:`~isaaclab_physx.sim.schemas.PhysxJointCfg`.
  * ``PhysxArticulationRootPropertiesCfg`` and ``ArticulationRootPropertiesCfg`` →
    :class:`~isaaclab_physx.sim.schemas.PhysxArticulationCfg`.
  * ``MeshCollisionPropertiesCfg`` → :class:`~isaaclab.sim.schemas.UsdPhysicsMeshCollisionCfg`;
    ``PhysxConvexHullPropertiesCfg`` / ``ConvexHullPropertiesCfg`` →
    :class:`~isaaclab_physx.sim.schemas.PhysxConvexHullCfg`;
    ``PhysxConvexDecompositionPropertiesCfg`` / ``ConvexDecompositionPropertiesCfg`` →
    :class:`~isaaclab_physx.sim.schemas.PhysxConvexDecompositionCfg`;
    ``PhysxTriangleMeshPropertiesCfg`` / ``TriangleMeshPropertiesCfg`` →
    :class:`~isaaclab_physx.sim.schemas.PhysxTriangleMeshCfg`;
    ``PhysxTriangleMeshSimplificationPropertiesCfg`` / ``TriangleMeshSimplificationPropertiesCfg`` →
    :class:`~isaaclab_physx.sim.schemas.PhysxTriangleMeshSimplificationCfg`;
    ``PhysxSDFMeshPropertiesCfg`` / ``SDFMeshPropertiesCfg`` →
    :class:`~isaaclab_physx.sim.schemas.PhysxSDFMeshCfg`.
  * ``OmniPhysicsDeformableBodyPropertiesCfg`` →
    :class:`~isaaclab.sim.schemas.OmniPhysicsDeformableBodyCfg`; ``PhysXDeformableBodyPropertiesCfg``
    → :class:`~isaaclab_physx.sim.schemas.PhysxDeformableBodyCfg` (plus
    :class:`~isaaclab_physx.sim.schemas.PhysxSurfaceDeformableBodyCfg` for surface deformables);
    ``PhysxDeformableBodyPropertiesCfg`` and ``DeformableBodyPropertiesCfg`` → all of them, in the
    ``volume_deformable_props`` or ``surface_deformable_props`` slot. The removed cfgs authored
    ``kinematic_enabled=False`` and ``solver_position_iteration_count=16`` by default; set them on
    the fragments to keep the authored USD identical.
  * ``PhysxFixedTendonPropertiesCfg`` / ``FixedTendonPropertiesCfg`` →
    :class:`~isaaclab_physx.sim.schemas.PhysxTendonAxisRootCfg` (and
    :class:`~isaaclab_physx.sim.schemas.PhysxTendonAxisCfg` for per-axis gearing);
    ``PhysxSpatialTendonPropertiesCfg`` / ``SpatialTendonPropertiesCfg`` →
    :class:`~isaaclab_physx.sim.schemas.PhysxTendonAttachmentRootCfg`. The ``Physx*`` tendon cfgs
    had no warning of their own, but their only consumers were the deprecated legacy tendon writers.

* Removed the ``define_deformable_body_properties`` and ``modify_deformable_body_properties``
  re-exports from :mod:`isaaclab_physx.sim.schemas`. Use
  :func:`~isaaclab.sim.schemas.apply_volume_deformable_properties` or
  :func:`~isaaclab.sim.schemas.apply_surface_deformable_properties`.
* Removed the ``RigidBodyMaterialCfg``, ``DeformableBodyMaterialCfg`` and
  ``SurfaceDeformableBodyMaterialCfg`` aliases deprecated in 3.1. Use
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg`,
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxDeformableBodyMaterialCfg` and
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxSurfaceDeformableBodyMaterialCfg`.
