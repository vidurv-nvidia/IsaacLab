* **Breaking:** Changed the spawner and :class:`~isaaclab.sim.converters.MeshConverterCfg` schema
  slots (``rigid_props``, ``collision_props``, ``mass_props``, ``mesh_collision_props``,
  ``articulation_props``, ``joint_drive_props``, ``fixed_tendons_props``, ``spatial_tendons_props``)
  to accept only schema fragments: a fragment, a list of fragments, or a mapping from target pattern
  to fragments. Any other value now raises ``TypeError`` at spawn time instead of being routed to a
  legacy writer. Replace a single cfg with the fragments listed for it in the 3.0 migration guide.
* **Breaking:** Changed :attr:`~isaaclab.sim.spawners.DeformableObjectSpawnerCfg.mass_props` to the
  same fragment union as the rigid-object spawners, since the single mass cfg it accepted was
  removed. Deformable bodies ignore ``UsdPhysics.MassAPI``; set their mass through
  :class:`~isaaclab.sim.schemas.OmniPhysicsDeformableBodyCfg` in a deformable slot.
* Changed the ``spawn_prims.py`` tutorial and the shared integration-test scene to author physics
  through schema fragments.
