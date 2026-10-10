isaaclab_physx.sim.schemas
==========================

.. automodule:: isaaclab_physx.sim.schemas

  PhysX schema fragments. Each fragment authors one PhysX-namespaced attribute group
  (``physx*:*``) and applies the corresponding ``Physx*API`` applied schema. Compose them with
  the engine-neutral fragments in :mod:`isaaclab.sim.schemas` in a spawner slot. See
  :ref:`schema-fragments` for the design.

  .. rubric:: Rigid body and joint

  .. autosummary::

    PhysxRigidBodyCfg
    PhysxJointCfg

  .. rubric:: Collision

  .. autosummary::

    PhysxCollisionCfg

  .. rubric:: Articulation root

  .. autosummary::

    PhysxArticulationCfg

  .. rubric:: Mesh collision (PhysX cooking)

  .. autosummary::

    PhysxConvexHullCfg
    PhysxConvexDecompositionCfg
    PhysxTriangleMeshCfg
    PhysxTriangleMeshSimplificationCfg
    PhysxSDFMeshCfg

  .. rubric:: Tendon

  .. autosummary::

    PhysxTendonAxisRootCfg
    PhysxTendonAxisCfg
    PhysxTendonAttachmentRootCfg

  .. rubric:: Deformable body

  .. autosummary::

    PhysxDeformableBodyCfg
    PhysxSurfaceDeformableBodyCfg

  .. rubric:: Functions

  .. autosummary::

    apply_physx_joint

.. currentmodule:: isaaclab_physx.sim.schemas

Rigid Body and Joint
--------------------

.. autoclass:: PhysxRigidBodyCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxJointCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autofunction:: apply_physx_joint

Collision
---------

.. autoclass:: PhysxCollisionCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Articulation Root
-----------------

.. autoclass:: PhysxArticulationCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Mesh Collision (PhysX cooking)
-------------------------------

.. autoclass:: PhysxConvexHullCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxConvexDecompositionCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxTriangleMeshCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxTriangleMeshSimplificationCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxSDFMeshCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Tendon
------

.. autoclass:: PhysxTendonAxisRootCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxTendonAxisCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxTendonAttachmentRootCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Deformable Body
---------------

.. autoclass:: PhysxDeformableBodyCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autoclass:: PhysxSurfaceDeformableBodyCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

The deformable family writers live in :mod:`isaaclab.sim.schemas`.
