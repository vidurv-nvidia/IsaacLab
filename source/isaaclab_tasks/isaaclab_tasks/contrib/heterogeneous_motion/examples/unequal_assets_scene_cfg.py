# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene import add as scene_add
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_assets import HUMANOID_28_CFG


def _robot_cfg() -> ArticulationCfg:
    return HUMANOID_28_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _platform_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.05)),
    )


def _barrier_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Barrier",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.25, 0.0, 0.35)),
    )


def _crate_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Crate",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.75, 0.0, 0.25)),
    )


def _add_global_assets(scene: InteractiveSceneCfg) -> InteractiveSceneCfg:
    scene.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    scene.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    return scene


@configclass
class UnequalAssetsCompoundSceneCfg(InteractiveSceneCfg):
    """Fixed logical layout slot using :class:`~isaaclab.sim.MultiAssetSpawnerCfg`."""

    robot: ArticulationCfg = _robot_cfg()
    layout: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Layout",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                    scale=(8.0, 8.0, 8.0),
                ),
                sim_utils.UsdFileCfg(
                    usd_path=(f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/factory_gear_base/factory_gear_base.usd"),
                    scale=(2.0, 4.0, 4.0),
                ),
            ],
            random_choice=False,
            rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True)],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.25)),
    )
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class UnequalAssetsInclusionSetSceneCfg(InteractiveSceneCfg):
    """Explicit legal combinations with one or three static layout assets."""

    clone_cfg: CloneCfg = CloneCfg(
        clone_combinations=[
            InclusionSet(assets=["robot", "platform"]),
            InclusionSet(assets=["robot", "platform", "barrier", "crate"]),
        ]
    )

    robot: ArticulationCfg = _robot_cfg()
    platform: RigidObjectCfg = _platform_cfg()
    barrier: RigidObjectCfg = _barrier_cfg()
    crate: RigidObjectCfg = _crate_cfg()
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class _SmallLayoutSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _robot_cfg()
    platform: RigidObjectCfg = _platform_cfg()


@configclass
class _LargeLayoutSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _robot_cfg()
    platform: RigidObjectCfg = _platform_cfg()
    barrier: RigidObjectCfg = _barrier_cfg()
    crate: RigidObjectCfg = _crate_cfg()


def _make_unequal_assets_scene_add_cfg(
    num_envs: int, env_spacing: float, replicate_physics: bool
) -> InteractiveSceneCfg:
    """Compose small and large source scenes into alternating legal layouts."""
    small = _SmallLayoutSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=replicate_physics,
    )
    large = _LargeLayoutSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=replicate_physics,
    )
    return _add_global_assets(scene_add(small, large))
