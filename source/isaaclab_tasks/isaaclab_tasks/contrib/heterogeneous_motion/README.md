# Heterogeneous Maze Motion Demonstrator

This customer-facing task overlay targets **Isaac Lab 3.0.0 Beta 2** (`VERSION` `3.0.0`), Python 3.12, and the PhysX backend. It demonstrates how an `InteractiveSceneCfg` clone plan selects layout-associated motion data without changing Isaac Lab core. It is intentionally a small technology demonstrator, not a general task framework.

The end-to-end task is `IsaacContrib-Heterogeneous-Maze-Motion-Direct`. It is a `DirectRLEnv`; there is no manager-based maze task. Its two physical layouts have one or two divider assets and route to `motions/maze_upper.csv` or `motions/maze_lower.csv` through a task-local lookup from ``ClonePlan.clone_mask.T``.

## Commands

From an Isaac Lab 3.0.0 checkout that contains this overlay, run the maintained validation entry point first:

```bash
./isaaclab.sh -p -m isaaclab_tasks.contrib.heterogeneous_motion.validation --num-envs 2 --steps 100 --device cuda:0
./isaaclab.sh -p -m isaaclab_tasks.contrib.heterogeneous_motion.validation --num-envs 4 --steps 100 --device cuda:0
```

The 2- and 4-environment checks use zero and random actions, validate finite signals and `[num_envs, 6]` policy observations, verify reset-stable physical-signature grouping and layout-to-motion routing, and check that the reference controller reaches the terminal sample. A scale smoke is deliberately one step:

```bash
./isaaclab.sh -p -m isaaclab_tasks.contrib.heterogeneous_motion.validation --num-envs 1000 --steps 1 --device cuda:0
```

At 1,000 environments, expect `clone_mask.shape == (8, 1000)` and exactly 500 environments in each layout group. Run it only in a known-good GPU/PhysX Isaac Lab installation.

Train and evaluate without checking generated artifacts into source:

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Heterogeneous-Maze-Motion-Direct --num_envs 128 agent.max_iterations=5
uv run isaaclab play --rl_library rsl_rl --task IsaacContrib-Heterogeneous-Maze-Motion-Direct --checkpoint /absolute/path/to/model_*.pt
```

No trained checkpoint is included: it is hardware-, seed-, and Isaac Sim-build-specific. Training writes logs and checkpoints under the runtime log directory; keep them outside this package and provide their exact command, seed, and checkpoint separately when sharing an evaluation.

## Registered task IDs

| Task ID | Purpose |
| --- | --- |
| `IsaacContrib-Heterogeneous-Maze-Motion-Direct` | Main DirectRLEnv maze, file-backed 2D motion paths, RSL-RL PPO config. |
| `IsaacContrib-Heterogeneous-Motion-Direct` | Legacy procedural direct construction example. |
| `IsaacContrib-Heterogeneous-Motion` | Legacy procedural manager-based example; not a maze. |
| `IsaacContrib-Heterogeneous-FixedSlots-MultiUsd-Direct` | Fixed logical layout slot using `MultiUsdFileCfg`; replace its sample asset paths with compound customer layout USDs. |
| `IsaacContrib-Heterogeneous-Unequal-Compound-Direct` | Fixed logical slots using `MultiAssetSpawnerCfg`. |
| `IsaacContrib-Heterogeneous-Unequal-InclusionSet-Direct` | Unequal named top-level assets using `CloneCfg` + `InclusionSet`. |
| `IsaacContrib-Heterogeneous-Unequal-SceneAdd-Direct` | Composition of existing `InteractiveSceneCfg` layouts using `isaaclab.scene.add`. |

## Directory inventory

- `layouts.py`, `scene_cfg.py`, `direct_env_cfg.py`, `direct_env.py`: task-owned layout table, maze layout, direct task, and clone-mask-to-motion routing.
- `motions/*.csv`, `motion.py`: packaged motion references and their deterministic generator/loader.
- `agents/rsl_rl_ppo_cfg.py`: compact PPO configuration for the maze.
- `examples/`: concise construction-pattern registrations. The two fixed-slot variants use `MultiAssetSpawnerCfg` and `MultiUsdFileCfg`; the latter is the compound-USD whole-layout pattern.
- `validation.py`: maintained 2/4/1000-environment PhysX validation entry point.
- `heterogeneous_environments_motion_references.md` at the repository root: construction and routing guide.
- `bundle.py`: creates the shareable overlay archive.

`bundle.py` includes this task directory, focused task tests, the package `pyproject.toml`, the guide, `VERSION`, and the repository BSD-3-Clause `LICENSE`. It does not modify or package Isaac Lab core, and it excludes Python caches, logs, checkpoints, and temporary files.
