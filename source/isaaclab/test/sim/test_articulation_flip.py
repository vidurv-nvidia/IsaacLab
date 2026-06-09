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


def test_articulation_properties_factory_returns_fragments():
    from isaaclab_physx.sim.schemas import ArticulationRootPropertiesCfg, PhysxArticulationCfg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frags = ArticulationRootPropertiesCfg(enabled_self_collisions=True, solver_position_iteration_count=8)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert any(isinstance(f, PhysxArticulationCfg) and f.solver_position_iteration_count == 8 for f in frags)


def test_articulation_factory_drops_fix_root_link():
    from isaaclab_physx.sim.schemas import ArticulationRootPropertiesCfg, PhysxArticulationCfg

    frags = ArticulationRootPropertiesCfg(fix_root_link=True, solver_position_iteration_count=4)
    assert all(not hasattr(f, "fix_root_link") for f in frags)  # not a fragment field
    assert any(isinstance(f, PhysxArticulationCfg) for f in frags)
