# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import warnings


def _assert_warns_deprecation(recorded) -> None:
    assert any(issubclass(x.category, DeprecationWarning) for x in recorded)


def test_mesh_collision_factory_returns_usd_fragment():
    from isaaclab_physx.sim.schemas import MeshCollisionPropertiesCfg

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frags = MeshCollisionPropertiesCfg(mesh_approximation_name="boundingCube")
    _assert_warns_deprecation(w)
    assert isinstance(frags, list)
    assert len(frags) == 1
    assert isinstance(frags[0], UsdPhysicsMeshCollisionCfg)
    assert frags[0].mesh_approximation_name == "boundingCube"


def test_convex_hull_factory_returns_fragments():
    from isaaclab_physx.sim.schemas import ConvexHullPropertiesCfg, PhysxConvexHullCfg

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frags = ConvexHullPropertiesCfg(hull_vertex_limit=32)
    _assert_warns_deprecation(w)
    usd = [f for f in frags if isinstance(f, UsdPhysicsMeshCollisionCfg)]
    cooking = [f for f in frags if isinstance(f, PhysxConvexHullCfg)]
    assert len(usd) == 1 and usd[0].mesh_approximation_name == "convexHull"
    assert len(cooking) == 1 and cooking[0].hull_vertex_limit == 32


def test_convex_decomposition_factory_returns_fragments():
    from isaaclab_physx.sim.schemas import ConvexDecompositionPropertiesCfg, PhysxConvexDecompositionCfg

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frags = ConvexDecompositionPropertiesCfg(max_convex_hulls=8, shrink_wrap=True)
    _assert_warns_deprecation(w)
    usd = [f for f in frags if isinstance(f, UsdPhysicsMeshCollisionCfg)]
    cooking = [f for f in frags if isinstance(f, PhysxConvexDecompositionCfg)]
    assert len(usd) == 1 and usd[0].mesh_approximation_name == "convexDecomposition"
    assert len(cooking) == 1
    assert cooking[0].max_convex_hulls == 8 and cooking[0].shrink_wrap is True


def test_triangle_mesh_factory_returns_fragments():
    from isaaclab_physx.sim.schemas import PhysxTriangleMeshCfg, TriangleMeshPropertiesCfg

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frags = TriangleMeshPropertiesCfg(weld_tolerance=0.01)
    _assert_warns_deprecation(w)
    usd = [f for f in frags if isinstance(f, UsdPhysicsMeshCollisionCfg)]
    cooking = [f for f in frags if isinstance(f, PhysxTriangleMeshCfg)]
    assert len(usd) == 1 and usd[0].mesh_approximation_name == "none"
    assert len(cooking) == 1 and abs(cooking[0].weld_tolerance - 0.01) < 1e-6


def test_triangle_mesh_simplification_factory_returns_fragments():
    from isaaclab_physx.sim.schemas import PhysxTriangleMeshSimplificationCfg, TriangleMeshSimplificationPropertiesCfg

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frags = TriangleMeshSimplificationPropertiesCfg(simplification_metric=0.7)
    _assert_warns_deprecation(w)
    usd = [f for f in frags if isinstance(f, UsdPhysicsMeshCollisionCfg)]
    cooking = [f for f in frags if isinstance(f, PhysxTriangleMeshSimplificationCfg)]
    assert len(usd) == 1 and usd[0].mesh_approximation_name == "meshSimplification"
    assert len(cooking) == 1 and abs(cooking[0].simplification_metric - 0.7) < 1e-6


def test_sdf_mesh_factory_returns_fragments():
    from isaaclab_physx.sim.schemas import PhysxSDFMeshCfg, SDFMeshPropertiesCfg

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frags = SDFMeshPropertiesCfg(sdf_resolution=128, sdf_margin=0.02)
    _assert_warns_deprecation(w)
    usd = [f for f in frags if isinstance(f, UsdPhysicsMeshCollisionCfg)]
    cooking = [f for f in frags if isinstance(f, PhysxSDFMeshCfg)]
    assert len(usd) == 1 and usd[0].mesh_approximation_name == "sdf"
    assert len(cooking) == 1
    assert cooking[0].sdf_resolution == 128 and abs(cooking[0].sdf_margin - 0.02) < 1e-6


def test_cooking_factory_without_tuning_fields_omits_cooking_fragment():
    """A bare cooking factory still yields the USD token fragment with the cooking variant's token."""
    from isaaclab_physx.sim.schemas import ConvexHullPropertiesCfg, PhysxConvexHullCfg

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        frags = ConvexHullPropertiesCfg()
    assert any(isinstance(f, UsdPhysicsMeshCollisionCfg) for f in frags)
    usd = next(f for f in frags if isinstance(f, UsdPhysicsMeshCollisionCfg))
    assert usd.mesh_approximation_name == "convexHull"
    # No tuning fields set -> no cooking fragment needs to be emitted.
    assert not any(isinstance(f, PhysxConvexHullCfg) for f in frags)
