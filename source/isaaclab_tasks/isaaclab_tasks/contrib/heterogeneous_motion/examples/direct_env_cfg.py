# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .scene_cfg import HeterogeneousMotionSceneCfg


@configclass
class HeterogeneousMotionDirectEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct heterogeneous motion-reference environment."""

    episode_length_s = 5.0
    decimation = 2
    action_space = 28
    observation_space = 85
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation, physics=PhysxCfg())
    scene: HeterogeneousMotionSceneCfg = HeterogeneousMotionSceneCfg(
        num_envs=2, env_spacing=4.0, replicate_physics=True
    )
