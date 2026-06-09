Changed
^^^^^^^

* **Breaking:** Deprecated :class:`CollisionPropertiesCfg` in favor of a fragment list
  ``[UsdPhysicsCollisionCfg(...), PhysxCollisionCfg(...)]``. The name now returns the equivalent
  fragments and emits a ``DeprecationWarning``; removal in 5.0.
