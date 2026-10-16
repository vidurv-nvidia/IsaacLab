# Unitree G1 velocity-tracking environments

This folder contains the flat-terrain, rough-terrain and VBD variants of the
Unitree G1 velocity-tracking task.

| Environment | Terrain | Default physics | Configuration | RSL-RL experiment |
| --- | --- | --- | --- | --- |
| Isaac-Velocity-Flat-G1 | Flat | Newton MJWarp | [flat_env_cfg.py](flat_env_cfg.py) | g1_flat |
| Isaac-Velocity-Rough-G1 | Rough | Newton MJWarp | [rough_env_cfg.py](rough_env_cfg.py) | g1_rough |
| Isaac-Velocity-Flat-G1-VBD | Flat | Newton VBD | [flat_vbd_env_cfg.py](flat_vbd_env_cfg.py) | g1_flat_vbd |

The environments reuse the existing G1_MINIMAL_CFG asset. Their registrations
are in [__init__.py](__init__.py), and their RSL-RL runner configurations are in
[agents/rsl_rl_ppo_cfg.py](agents/rsl_rl_ppo_cfg.py).

Run training from the Isaac Lab repository root, choosing an environment from
the table:

```bash
uv run --frozen isaaclab train \
    --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 \
    --num_envs 4096 --device cuda:0
```

The VBD variant inherits the flat task's asset, observations, actions and
rewards, then selects its own solver configuration and physics timestep.
It has a separate task ID and training experiment; selecting the original
flat or rough task keeps that task's original configuration.

See the [G1 VBD guide](../../../../../../../docs/g1_vbd.md) for VBD installation
requirements, solver and actuator settings, training, playback, validation
and upstream integration notes.
