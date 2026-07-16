# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets import HUMANOID_28_CFG


@configclass
class HeterogeneousMotionSceneCfg(InteractiveSceneCfg):
    """Two course variants and one shared humanoid."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    course = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Course",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 0.1),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
                ),
                sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 0.4),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.8)),
                ),
            ],
            random_choice=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.5, 0.0, 0.2)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
