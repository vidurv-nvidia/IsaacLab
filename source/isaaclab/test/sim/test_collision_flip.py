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


def test_collision_properties_factory_returns_fragments():
    from isaaclab_physx.sim.schemas import CollisionPropertiesCfg, PhysxCollisionCfg

    from isaaclab.sim.schemas import UsdPhysicsCollisionCfg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frags = CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.02, rest_offset=0.0)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert any(isinstance(f, UsdPhysicsCollisionCfg) and f.collision_enabled is True for f in frags)
    assert any(isinstance(f, PhysxCollisionCfg) and abs(f.contact_offset - 0.02) < 1e-9 for f in frags)
