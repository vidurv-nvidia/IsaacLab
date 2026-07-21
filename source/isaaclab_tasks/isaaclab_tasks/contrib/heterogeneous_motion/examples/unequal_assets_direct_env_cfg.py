# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from .direct_env_cfg import HeterogeneousMotionDirectEnvCfg
from .unequal_assets_scene_cfg import (
    UnequalAssetsCompoundSceneCfg,
    UnequalAssetsInclusionSetSceneCfg,
    _make_unequal_assets_scene_add_cfg,
)


@configclass
class UnequalAssetsCompoundDirectEnvCfg(HeterogeneousMotionDirectEnvCfg):
    """Direct environment selecting compound USD layouts with unequal child counts."""

    scene: UnequalAssetsCompoundSceneCfg = UnequalAssetsCompoundSceneCfg(
        num_envs=2, env_spacing=4.0, replicate_physics=True
    )


@configclass
class UnequalAssetsInclusionSetDirectEnvCfg(HeterogeneousMotionDirectEnvCfg):
    """Direct environment using explicit clone inclusion sets."""

    scene: UnequalAssetsInclusionSetSceneCfg = UnequalAssetsInclusionSetSceneCfg(
        num_envs=2, env_spacing=4.0, replicate_physics=True
    )


@configclass
class UnequalAssetsSceneAddDirectEnvCfg(HeterogeneousMotionDirectEnvCfg):
    """Direct environment composing unequal source scenes with :func:`~isaaclab.scene.add`."""

    scene: InteractiveSceneCfg = _make_unequal_assets_scene_add_cfg(num_envs=2, env_spacing=4.0, replicate_physics=True)
