# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg


def test_g1_rough_newton_has_sufficient_constraint_capacity():
    env_cfg = G1RoughEnvCfg()

    solver_cfg = env_cfg.sim.physics.newton_mjwarp.solver_cfg

    assert isinstance(solver_cfg, MJWarpSolverCfg)
    assert solver_cfg.njmax == 300


def test_g1_vbd_task_is_opt_in_and_retains_policy_interface():
    """Selecting VBD must not change the existing G1 task or policy dimensions."""
    from isaaclab_newton.physics import VBDSolverCfg

    from isaaclab_tasks.utils import resolve_task_config

    standard, standard_agent = resolve_task_config("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point", overrides=[])
    vbd, vbd_agent = resolve_task_config("Isaac-Velocity-Flat-G1-VBD", "rsl_rl_cfg_entry_point", overrides=[])

    assert isinstance(standard.sim.physics.solver_cfg, MJWarpSolverCfg)
    assert isinstance(vbd.sim.physics.solver_cfg, VBDSolverCfg)
    assert standard.scene.robot == vbd.scene.robot
    assert standard.observations == vbd.observations
    assert standard.actions == vbd.actions
    assert standard.sim.dt * standard.decimation == vbd.sim.dt * vbd.decimation
    assert standard_agent.actor == vbd_agent.actor
    assert standard_agent.critic == vbd_agent.critic
    assert standard_agent.experiment_name != vbd_agent.experiment_name
