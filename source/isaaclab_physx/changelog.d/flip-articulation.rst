Changed
^^^^^^^

* **Breaking:** Turned :class:`~isaaclab_physx.sim.schemas.ArticulationRootPropertiesCfg` into a
  deprecated factory that returns an articulation-root fragment list instead of a config object.
  Migrate ``articulation_props=ArticulationRootPropertiesCfg(...)`` to
  ``articulation_props=[PhysxArticulationCfg(...)]`` using
  :class:`~isaaclab_physx.sim.schemas.PhysxArticulationCfg`. The ``fix_root_link`` argument is no
  longer a fragment field; it is now a spawner-level flag, so set it on the spawner cfg that owns
  the prim instead (e.g. :attr:`isaaclab.sim.spawners.from_files.UsdFileCfg.fix_root_link`, or pass
  ``fix_root_link=`` to :func:`~isaaclab.sim.schemas.apply_articulation_root_properties`). Removal in 5.0.
