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


def test_joint_drive_properties_factory_returns_fragments():
    from isaaclab_physx.sim.schemas import JointDrivePropertiesCfg, PhysxJointCfg

    from isaaclab.sim.schemas import UsdPhysicsDriveCfg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frags = JointDrivePropertiesCfg(stiffness=10.0, damping=0.1, max_joint_velocity=5.0)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert any(isinstance(f, UsdPhysicsDriveCfg) and f.stiffness == 10.0 for f in frags)
    assert any(isinstance(f, PhysxJointCfg) and f.max_joint_velocity == 5.0 for f in frags)
