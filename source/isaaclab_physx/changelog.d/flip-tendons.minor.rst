Changed
^^^^^^^

* **Breaking:** Turned :func:`~isaaclab_physx.sim.schemas.FixedTendonPropertiesCfg` and
  :func:`~isaaclab_physx.sim.schemas.SpatialTendonPropertiesCfg` into deprecation factories
  that emit a ``DeprecationWarning`` and return the equivalent fragment list
  (``[PhysxFixedTendonCfg(...)]`` / ``[PhysxSpatialTendonCfg(...)]``). Pass a fragment list
  directly instead, e.g. ``fixed_tendons_props=[PhysxFixedTendonCfg(...)]`` (import
  :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonCfg` /
  :class:`~isaaclab_physx.sim.schemas.PhysxSpatialTendonCfg` from
  :mod:`isaaclab_physx.sim.schemas`). The legacy factories are scheduled for removal in 5.0.
