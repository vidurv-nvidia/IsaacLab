# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from .layouts import LAYOUTS


def _agent_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Agent",
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.3, 0.0, 0.2)),
    )


def _wall_cfg(
    prim_name: str,
    position: tuple[float, float],
    size: tuple[float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=(size[0], size[1], 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(position[0], position[1], 0.5)),
    )


@configclass
class HeterogeneousMazeSceneCfg(InteractiveSceneCfg):
    """Two sparse block mazes with unequal asset counts."""

    clone_cfg: CloneCfg = CloneCfg(clone_combinations=[layout.inclusion for layout in LAYOUTS])

    agent: RigidObjectCfg = _agent_cfg()

    top_wall: RigidObjectCfg = _wall_cfg("TopWall", (0.0, 2.1), (6.2, 0.2), (0.35, 0.35, 0.35))
    bottom_wall: RigidObjectCfg = _wall_cfg("BottomWall", (0.0, -2.1), (6.2, 0.2), (0.35, 0.35, 0.35))
    left_wall: RigidObjectCfg = _wall_cfg("LeftWall", (-3.0, 0.0), (0.2, 4.2), (0.35, 0.35, 0.35))
    right_wall: RigidObjectCfg = _wall_cfg("RightWall", (3.0, 0.0), (0.2, 4.2), (0.35, 0.35, 0.35))

    upper_divider: RigidObjectCfg = _wall_cfg("UpperRouteDivider", (0.0, -0.75), (0.35, 2.7), (0.15, 0.55, 0.2))
    lower_divider_lower: RigidObjectCfg = _wall_cfg(
        "LowerRouteDividerLower", (0.0, -0.05), (0.45, 1.1), (0.15, 0.3, 0.75)
    )
    lower_divider_upper: RigidObjectCfg = _wall_cfg(
        "LowerRouteDividerUpper", (0.0, 1.25), (0.3, 1.7), (0.15, 0.3, 0.75)
    )

    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
