* Removed the single schema cfgs deprecated in 3.1. Compose the schema fragments in the spawner slot
  instead (the 3.0 migration guide lists the complete fragment set for every class):

  * ``RigidBodyBaseCfg`` → :class:`~isaaclab.sim.schemas.UsdPhysicsRigidBodyCfg` plus
    :class:`~isaaclab_physx.sim.schemas.PhysxRigidBodyCfg` for ``disable_gravity``.
  * ``CollisionBaseCfg`` → :class:`~isaaclab.sim.schemas.UsdPhysicsCollisionCfg` plus
    :class:`~isaaclab_physx.sim.schemas.PhysxCollisionCfg` for ``contact_offset`` / ``rest_offset``;
    its nested ``mesh_collision_property`` → the spawner's ``mesh_collision_props`` slot.
  * ``MassPropertiesCfg`` → :class:`~isaaclab.sim.schemas.MassCfg`.
  * ``JointDriveBaseCfg`` → :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` plus
    :class:`~isaaclab_physx.sim.schemas.PhysxJointCfg` for ``max_joint_velocity``; its
    ``ensure_drives_exist`` → the spawner's ``ensure_drives_exist`` field.
  * ``ArticulationRootBaseCfg`` → :class:`~isaaclab_physx.sim.schemas.PhysxArticulationCfg`; its
    ``fix_root_link`` → the spawner's ``fix_root_link`` field.
  * ``MeshCollisionBaseCfg``, ``BoundingCubePropertiesCfg`` and ``BoundingSpherePropertiesCfg`` →
    :class:`~isaaclab.sim.schemas.UsdPhysicsMeshCollisionCfg` with the matching
    ``mesh_approximation_name``, in the ``mesh_collision_props`` slot.
  * ``DeformableBodyPropertiesBaseCfg`` → a :class:`~isaaclab.sim.schemas.DeformableBodyFragment`
    subclass, such as :class:`~isaaclab.sim.schemas.OmniPhysicsDeformableBodyCfg`.

* Removed the ``define_*_properties`` / ``modify_*_properties`` schema writers deprecated in 3.1:
  ``define_rigid_body_properties`` / ``modify_rigid_body_properties`` →
  :func:`~isaaclab.sim.schemas.apply_rigid_body_properties`,
  ``define_collision_properties`` / ``modify_collision_properties`` →
  :func:`~isaaclab.sim.schemas.apply_collision_properties`,
  ``define_mass_properties`` / ``modify_mass_properties`` →
  :func:`~isaaclab.sim.schemas.apply_mass_properties`,
  ``define_articulation_root_properties`` / ``modify_articulation_root_properties`` →
  :func:`~isaaclab.sim.schemas.apply_articulation_root_properties`,
  ``modify_joint_drive_properties`` → :func:`~isaaclab.sim.schemas.apply_joint_drive_properties`,
  ``define_mesh_collision_properties`` / ``modify_mesh_collision_properties`` →
  :func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`,
  ``modify_fixed_tendon_properties`` / ``modify_spatial_tendon_properties`` →
  :func:`~isaaclab.sim.schemas.apply_fixed_tendon_properties` /
  :func:`~isaaclab.sim.schemas.apply_spatial_tendon_properties`, and
  ``define_deformable_body_properties`` / ``modify_deformable_body_properties`` →
  :func:`~isaaclab.sim.schemas.apply_volume_deformable_properties` or
  :func:`~isaaclab.sim.schemas.apply_surface_deformable_properties` (``create_if_missing=True``
  replaces ``define_*``).
* Removed the ``deformable_props`` spawner field deprecated in 3.1. Use ``volume_deformable_props``,
  or ``surface_deformable_props`` when the physics material is a surface deformable material; an
  empty list creates the body with backend defaults.
* Removed ``PHYSX_MESH_COLLISION_CFGS`` and ``USD_MESH_COLLISION_CFGS`` from
  :mod:`isaaclab.sim.schemas`. They listed only the removed mesh-collision cfgs; use the
  mesh-collision fragments (subclasses of :class:`~isaaclab.sim.schemas.MeshCollisionFragment`).
* Removed the ``isaaclab.sim`` / ``isaaclab.sim.schemas`` / ``isaaclab.sim.spawners.materials``
  forwarding entries for the removed backend cfgs and for the ``RigidBodyMaterialCfg``,
  ``DeformableBodyMaterialCfg`` and ``SurfaceDeformableBodyMaterialCfg`` aliases. Import
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg`,
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxDeformableBodyMaterialCfg` and
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxSurfaceDeformableBodyMaterialCfg` instead.
* Removed ``spawn_rigid_body_material_from_fragments``, deprecated in 3.1. Use
  :func:`~isaaclab.sim.spawners.materials.spawn_physics_material_from_fragments`.
* Removed the ``concepts/schema_cfgs`` documentation page; :ref:`schema-fragments` now documents
  where each fragment lives.
