# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_assets import HUMANOID_28_CFG


@configclass
class FixedSlotsCompoundUsdSceneCfg(InteractiveSceneCfg):
    """Fixed logical slots with a :class:`~isaaclab.sim.MultiUsdFileCfg` layout variant.

    Each path represents one complete layout beneath the single ``Layout`` slot. The
    supplied Isaac assets keep the example runnable; customer layouts may replace them
    with compound USDs containing different internal collider hierarchies.
    """

    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    layout: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Layout",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=[
                f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/factory_gear_base/factory_gear_base.usd",
            ],
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.25)),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
