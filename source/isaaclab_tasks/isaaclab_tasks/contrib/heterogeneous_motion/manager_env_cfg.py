# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from . import motion_reference as motion_mdp
from .scene_cfg import HeterogeneousMotionSceneCfg


@configclass
class ActionsCfg:
    """Joint-position actions shared by both motion references."""

    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Policy observations including the selected reference and combination ID."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_position = ObsTerm(func=mdp.joint_pos_rel)
        joint_velocity = ObsTerm(func=mdp.joint_vel_rel)
        motion_reference = ObsTerm(func=motion_mdp.manager_reference_joint_positions)
        combination_id = ObsTerm(func=motion_mdp.manager_combination_id)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Reset the robot to its configured default state."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class RewardsCfg:
    """Track the reference selected by the environment's clone combination."""

    motion_tracking = RewTerm(func=motion_mdp.manager_motion_tracking_reward, weight=1.0)


@configclass
class TerminationsCfg:
    """End each episode at the configured time limit."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class HeterogeneousMotionManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the manager-based heterogeneous motion-reference environment."""

    scene: HeterogeneousMotionSceneCfg = HeterogeneousMotionSceneCfg(
        num_envs=2, env_spacing=4.0, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim = SimulationCfg(dt=1.0 / 120.0, render_interval=self.decimation, physics=PhysxCfg())
