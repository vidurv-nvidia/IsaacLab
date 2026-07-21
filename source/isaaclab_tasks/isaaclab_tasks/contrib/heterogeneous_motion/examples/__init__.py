# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction-pattern examples; the manager task is legacy and procedural."""

import gymnasium as gym


gym.register(
    id="IsaacContrib-Heterogeneous-Motion-Direct",
    entry_point=f"{__name__}.direct_env:HeterogeneousMotionDirectEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.direct_env_cfg:HeterogeneousMotionDirectEnvCfg"},
)

gym.register(
    id="IsaacContrib-Heterogeneous-FixedSlots-MultiUsd-Direct",
    entry_point=f"{__name__}.direct_env:HeterogeneousMotionDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.fixed_slots_direct_env_cfg:FixedSlotsCompoundUsdDirectEnvCfg"
        )
    },
)

gym.register(
    id="IsaacContrib-Heterogeneous-Motion",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.manager_env_cfg:HeterogeneousMotionManagerEnvCfg"},
)

gym.register(
    id="IsaacContrib-Heterogeneous-Unequal-Compound-Direct",
    entry_point=f"{__name__}.unequal_assets_direct_env:HeterogeneousUnequalAssetsDirectEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.unequal_assets_direct_env_cfg:UnequalAssetsCompoundDirectEnvCfg"},
)

gym.register(
    id="IsaacContrib-Heterogeneous-Unequal-InclusionSet-Direct",
    entry_point=f"{__name__}.unequal_assets_direct_env:HeterogeneousUnequalAssetsDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unequal_assets_direct_env_cfg:UnequalAssetsInclusionSetDirectEnvCfg"
    },
)

gym.register(
    id="IsaacContrib-Heterogeneous-Unequal-SceneAdd-Direct",
    entry_point=f"{__name__}.unequal_assets_direct_env:HeterogeneousUnequalAssetsDirectEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.unequal_assets_direct_env_cfg:UnequalAssetsSceneAddDirectEnvCfg"},
)
