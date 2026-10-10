isaaclab.sim.schemas
====================

.. automodule:: isaaclab.sim.schemas

  .. rubric:: Schema fragments

  A fragment mirrors exactly one USD applied schema and writes into a single attribute
  namespace. The family writers below dispatch lists of fragments to the prims matched by
  a target expression. See :ref:`schema-fragments` for the concept and the spawner-level
  usage. Backend fragments live in :mod:`isaaclab_physx.sim.schemas` and
  :mod:`isaaclab_newton.sim.schemas`.

  .. autosummary::

    SchemaFragment
    RigidBodyFragment
    CollisionFragment
    MassFragment
    ArticulationRootFragment
    JointDriveFragment
    MeshCollisionFragment
    FixedTendonFragment
    SpatialTendonFragment
    UsdPhysicsRigidBodyCfg
    UsdPhysicsCollisionCfg
    UsdPhysicsDriveCfg
    UsdPhysicsMeshCollisionCfg
    MassCfg
    DeformableBodyFragment
    OmniPhysicsDeformableBodyCfg

  .. rubric:: Fragment writers

  .. autosummary::

    apply_rigid_body_properties
    apply_collision_properties
    apply_mass_properties
    apply_articulation_root_properties
    apply_joint_drive_properties
    apply_mesh_collision_properties
    apply_fixed_tendon_properties
    apply_spatial_tendon_properties
    apply_namespaced
    apply_drive
    apply_mesh_collision
    apply_volume_deformable_properties
    apply_surface_deformable_properties

  .. rubric:: Functions

  .. autosummary::

    activate_contact_sensors
    define_deformable_curve_properties

Schema Fragments
----------------

.. autoclass:: SchemaFragment
    :members:
    :exclude-members: __init__

.. autoclass:: RigidBodyFragment
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: CollisionFragment
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: MassFragment
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: ArticulationRootFragment
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: JointDriveFragment
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: MeshCollisionFragment
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: FixedTendonFragment
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: SpatialTendonFragment
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: UsdPhysicsRigidBodyCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: UsdPhysicsCollisionCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: UsdPhysicsDriveCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: UsdPhysicsMeshCollisionCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: MassCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: DeformableBodyFragment
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: OmniPhysicsDeformableBodyCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autofunction:: apply_rigid_body_properties
.. autofunction:: apply_collision_properties
.. autofunction:: apply_mass_properties
.. autofunction:: apply_articulation_root_properties
.. autofunction:: apply_joint_drive_properties
.. autofunction:: apply_mesh_collision_properties
.. autofunction:: apply_fixed_tendon_properties
.. autofunction:: apply_spatial_tendon_properties
.. autofunction:: apply_namespaced
.. autofunction:: apply_drive
.. autofunction:: apply_mesh_collision
.. autofunction:: apply_volume_deformable_properties
.. autofunction:: apply_surface_deformable_properties

Contact Sensors
---------------

.. autofunction:: activate_contact_sensors

Tendon
------

PhysX tendon schemas are tuned through the tendon fragments in :mod:`isaaclab_physx.sim.schemas`.
Newton's MuJoCo solver also supports tendons;
:class:`~isaaclab_newton.sim.schemas.MujocoFixedTendonCfg` tunes fixed-tendon spring stiffness and damping.

Position limits specify a range of the accumulated tendon coordinate on both backends. Their force response
uses different parameters: PhysX uses force stiffness and shares tendon damping with the limit, while
MuJoCo uses separate ``solreflimit`` and ``solimplimit`` parameters. Newton already
`converts force gains for joint limits
<https://github.com/newton-physics/newton/blob/v1.6.0/newton/_src/solvers/mujoco/kernels.py#L2622-L2732>`_
using inverse inertia and impedance. The corresponding tendon conversion is not implemented, so Isaac Lab's
shared tendon limit-stiffness API still raises :class:`NotImplementedError`. This is an implementation gap;
MuJoCo supports stiffness/damping through its
`solver parameters <https://mujoco.readthedocs.io/en/stable/modeling.html#reference>`_.

Deformable Curve
----------------

.. autofunction:: define_deformable_curve_properties
