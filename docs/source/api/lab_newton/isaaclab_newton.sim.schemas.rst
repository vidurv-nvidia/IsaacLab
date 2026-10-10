isaaclab_newton.sim.schemas
===========================

.. automodule:: isaaclab_newton.sim.schemas

  Newton schema fragments. Each fragment authors Newton-namespaced attributes (``newton:*``)
  or attributes for Newton's MuJoCo solver (``mjc:*``). Compose them with the engine-neutral
  fragments in :mod:`isaaclab.sim.schemas` in a spawner slot. See :ref:`schema-fragments`
  for the design.

  .. rubric:: Newton (``newton:*``)

  .. autosummary::

    NewtonCollisionCfg
    NewtonMeshCollisionCfg
    NewtonSDFCollisionCfg
    NewtonArticulationCfg

  .. rubric:: MuJoCo solver (``mjc:*``)

  .. autosummary::

    MujocoRigidBodyCfg
    MujocoJointCfg
    MujocoCollisionCfg
    MujocoFixedTendonCfg

  .. rubric:: Material

  .. autosummary::

    NewtonMaterialPropertiesCfg

  .. rubric:: Functions

  .. autosummary::

    apply_mujoco_collision
    apply_mujoco_fixed_tendon
    apply_mujoco_joint

.. currentmodule:: isaaclab_newton.sim.schemas

Rigid Body
----------

.. autoclass:: MujocoRigidBodyCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Joint
-----

.. autoclass:: MujocoJointCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autofunction:: apply_mujoco_joint

Collision
---------

.. autoclass:: NewtonCollisionCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: MujocoCollisionCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autofunction:: apply_mujoco_collision

Mesh Collision
--------------

.. autoclass:: NewtonMeshCollisionCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: NewtonSDFCollisionCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Articulation Root
-----------------

.. autoclass:: NewtonArticulationCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Tendon
------

.. autoclass:: MujocoFixedTendonCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autofunction:: apply_mujoco_fixed_tendon

Material
--------

.. autoclass:: NewtonMaterialPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__
