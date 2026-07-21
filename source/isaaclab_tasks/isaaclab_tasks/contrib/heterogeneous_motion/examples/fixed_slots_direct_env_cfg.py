# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from .direct_env_cfg import HeterogeneousMotionDirectEnvCfg
from .fixed_slots_scene_cfg import FixedSlotsCompoundUsdSceneCfg


@configclass
class FixedSlotsCompoundUsdDirectEnvCfg(HeterogeneousMotionDirectEnvCfg):
    """Legacy direct task with one multi-USD whole-layout logical slot."""

    scene: InteractiveSceneCfg = FixedSlotsCompoundUsdSceneCfg(
        num_envs=2,
        env_spacing=4.0,
        replicate_physics=True,
    )
