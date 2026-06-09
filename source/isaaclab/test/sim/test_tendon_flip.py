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


def test_fixed_tendon_properties_cfg_is_deprecation_factory():
    """``FixedTendonPropertiesCfg(...)`` warns once and returns ``[PhysxFixedTendonCfg(...)]``."""
    from isaaclab_physx.sim.schemas import FixedTendonPropertiesCfg, PhysxFixedTendonCfg

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = FixedTendonPropertiesCfg(stiffness=10.0, damping=0.1)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected one DeprecationWarning, got {len(deprecations)}"
    assert "5.0" in str(deprecations[0].message)

    assert isinstance(result, list) and len(result) == 1
    frag = result[0]
    assert isinstance(frag, PhysxFixedTendonCfg)
    assert frag.stiffness == 10.0
    assert frag.damping == 0.1


def test_spatial_tendon_properties_cfg_is_deprecation_factory():
    """``SpatialTendonPropertiesCfg(...)`` warns once and returns ``[PhysxSpatialTendonCfg(...)]``."""
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg, SpatialTendonPropertiesCfg

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = SpatialTendonPropertiesCfg(stiffness=20.0, limit_stiffness=2.0)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected one DeprecationWarning, got {len(deprecations)}"
    assert "5.0" in str(deprecations[0].message)

    assert isinstance(result, list) and len(result) == 1
    frag = result[0]
    assert isinstance(frag, PhysxSpatialTendonCfg)
    assert frag.stiffness == 20.0
    assert frag.limit_stiffness == 2.0
