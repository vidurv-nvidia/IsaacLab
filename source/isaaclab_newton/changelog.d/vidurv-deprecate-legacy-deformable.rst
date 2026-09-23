Deprecated
^^^^^^^^^^

* Deprecated :class:`~isaaclab_newton.sim.schemas.NewtonDeformableBodyPropertiesCfg` in favor of
  :class:`~isaaclab_newton.sim.schemas.NewtonDeformableBodyCfg`. It now raises a
  ``DeprecationWarning`` on instantiation and will be removed in 3.2. Pass the fragment in the
  spawner's ``surface_deformable_props`` slot when ``physics_material`` is a surface deformable
  material and in ``volume_deformable_props`` otherwise, which is the type the legacy
  ``deformable_props`` field derived. The active physics backend now selects Newton's deformable
  schemas; the legacy cfg selected them itself.
