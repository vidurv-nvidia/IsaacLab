* **Breaking:** Changed the ``Isaac-Lift-Soft-Franka`` and ``Isaac-Lift-Cloth-Franka`` deformable presets
  (and their ``-Camera`` variants) to author the deformable body through the schema-fragment slots
  instead of the removed ``deformable_props`` field. The PhysX preset now passes
  ``[OmniPhysicsDeformableBodyCfg(kinematic_enabled=False), PhysxDeformableBodyCfg(solver_position_iteration_count=16)]``
  in ``volume_deformable_props`` (soft beam) or ``surface_deformable_props`` (cloth), and the Newton
  preset passes an empty slot. The authored USD is unchanged. Nested CLI overrides must target the
  slot element instead, e.g. replace
  ``env.scene.deformable.spawn.deformable_props.self_collision=true`` with
  ``env.scene.deformable.spawn.volume_deformable_props.1.self_collision=true`` for the soft beam and
  ``env.scene.deformable.spawn.surface_deformable_props.1.self_collision=true`` for the cloth
  (``.0.`` addresses the :class:`~isaaclab.sim.schemas.OmniPhysicsDeformableBodyCfg` fields).
