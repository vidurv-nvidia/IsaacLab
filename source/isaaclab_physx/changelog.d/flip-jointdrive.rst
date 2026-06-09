Changed
^^^^^^^

* **Breaking:** Turned :func:`~isaaclab_physx.sim.schemas.JointDrivePropertiesCfg` into a
  deprecation factory that returns a joint-drive fragment list instead of a config object. It now
  emits a :class:`DeprecationWarning` and splits its keyword arguments into a
  :class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` fragment (USD ``DriveAPI`` fields:
  ``drive_type``, ``max_force`` / ``max_effort``, ``stiffness``, ``damping``) and, when backend
  fields (``max_joint_velocity`` / ``max_velocity``) are present, a
  :class:`~isaaclab_physx.sim.schemas.PhysxJointCfg` fragment. Migrate by passing a fragment list
  directly, e.g. ``joint_drive_props=[UsdPhysicsDriveCfg(drive_type="force"),
  PhysxJointCfg(max_joint_velocity=5.0)]``. The ``ensure_drives_exist`` argument is no longer a
  fragment field; it is now a writer flag, set via the spawner cfg's ``ensure_drives_exist`` (or
  passed to :func:`~isaaclab.sim.schemas.apply_joint_drive_properties`). Scheduled for removal in
  5.0.
