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


def test_mass_properties_factory_returns_fragments():
    from isaaclab.sim.schemas import MassCfg, MassPropertiesCfg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frags = MassPropertiesCfg(mass=2.0, density=100.0)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert len(frags) == 1 and isinstance(frags[0], MassCfg)
    assert frags[0].mass == 2.0 and frags[0].density == 100.0
