# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""File-backed heterogeneous motion-reference environment."""

import gymnasium as gym


gym.register(
    id="IsaacContrib-Heterogeneous-Maze-Motion-Direct",
    entry_point=f"{__name__}.direct_env:HeterogeneousMazeDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env_cfg:HeterogeneousMazeDirectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HeterogeneousMazePPORunnerCfg",
    },
)
