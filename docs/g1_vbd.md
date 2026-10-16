# G1 locomotion with Newton VBD

This branch adds **Isaac-Velocity-Flat-G1-VBD**, a separate flat-terrain velocity
tracking environment for the existing Unitree G1 asset. It uses Newton's sparse
rigid articulation solver with BDF2 integration and compliant ALM. The standard
Isaac-Velocity-Flat-G1 task keeps its original physics configuration.

## Install

The validated configuration uses Python 3.12, Warp 1.17.0, PyTorch 2.11.0 and
RSL-RL 5.5.1 on Linux with an NVIDIA RTX PRO 6000 Blackwell GPU. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and a CUDA-capable
driver compatible with the locked packages, then run:

```bash
git clone --branch codex/g1-vbd-flat --single-branch https://github.com/vidurv-nvidia/IsaacLab.git
cd IsaacLab
uv sync --frozen
```

The manifest and lockfile pin
[Newton 0534e713](https://github.com/skulkar22/newton/commit/0534e713d176a029d3f94f2b44aa2fbe2205400b)
from the dev/rigid-vbd-integration development branch. Keep this pin: other Newton
revisions can change solver options, contact sensing and physical behavior.
This is a dedicated preview branch; the dependency pin applies to the whole
checkout. Newton training uses the Kit-less runtime and does not require launching
Isaac Sim.

The environment reuses Isaac Lab's G1_MINIMAL_CFG and its published
Robots/Unitree/G1/g1_minimal.usd asset. The first launch requires network access
to the Isaac Lab asset server to resolve the USD and mesh dependencies. No
workstation-local model paths, H2 assets, or separate H2 changes are required.
Asset licensing remains governed by the source asset's terms.

## Train

Use the existing RSL-RL training entrypoint. On a workstation with multiple GPUs,
this example selects physical GPU 1; the process then addresses it as cuda:0.
Choose a different mask or omit it on a single-GPU machine.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --frozen isaaclab train \
    --rl_library rsl_rl --task Isaac-Velocity-Flat-G1-VBD \
    --num_envs 4096 --device cuda:0 --seed 42
```

The task's runner defaults to 1,500 PPO updates, 24 steps per environment per
update, and a checkpoint every 50 updates. Logs and checkpoints are written
under logs/rsl_rl/g1_flat_vbd/. A smaller launch check is:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --frozen isaaclab train \
    --rl_library rsl_rl --task Isaac-Velocity-Flat-G1-VBD \
    --num_envs 64 --device cuda:0 --max_iterations 5 --logger tensorboard
```

A short check verifies execution; it is too small to learn walking from scratch.

## Play and record

Use an absolute path to a checkpoint produced by training:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --frozen --extra video isaaclab play \
    --rl_library rsl_rl --task Isaac-Velocity-Flat-G1-VBD \
    --checkpoint /absolute/path/to/model_1499.pt \
    --num_envs 1 --device cuda:0 --seed 42 \
    --video newton_gl --video_length 1000
```

The video extra installs the optional moviepy encoder. This records 20 seconds
of simulation with the native Newton GL renderer.
Video recording requires a working OpenGL context; on a server, configure the
renderer/display before recording. See the
[recording guide](source/how-to/record_video.rst) for output location and renderer
options. Training itself does not need a graphics context. Playback uses the
native inference policy and the task's normal play-mode settings, including
disabled observation noise and training pushes. Do not add --deterministic:
this VBD configuration does not support Isaac Lab's deterministic solver mode.
Checkpoints and videos are generated artifacts and are not bundled in Git.

## Solver and actuator settings

| Setting | Value |
| --- | --- |
| Physics | Newton VBD, block_sparse_joints |
| Time integration | BDF2 |
| Physics timestep | 0.004 s, one substep |
| Policy/action interval | 0.020 s, decimation 5 |
| Solver iterations | 7 |
| Articulation relaxation | 0.65 |
| Compliant ALM | Enabled |
| Structural linear/angular stiffness | 1,000,000 N/m and N m/rad |
| Structural linear/angular damping | 500 N s/m and N m s/rad |
| Rigid contact capacity | 256 contacts per body |
| Axis-support armature / limit rho floor | Enabled / enabled |
| Contact sensor update | Every 0.004 s |

Structural ALM stiffness and damping constrain the articulation. They are
separate from the motor drive gains. The environment preserves the existing
G1 asset's masses, inertia, joint limits, armature, initial pose and actuator
configuration, as well as the flat task's observations, actions, rewards and
randomization.

Actuation uses **VBD internal joint drives**, through the asset's existing
ImplicitActuatorCfg groups; it does not use Newton actuator-library instances.

| Group | Position gain | Velocity gain | Armature |
| --- | --- | --- | --- |
| Hip / knee / torso | 150–200 | 5 | 0.01 |
| Ankle | 20 | 2 | 0.01 |
| Arm / finger | 40 | 10 | 0.01 / 0.001 |

These are authored asset values; the inherited task retains its training
randomization. Do not treat configured effort limits as validated motor torque
saturation on the internal VBD drive path.

The Isaac Lab contact bridge requests native CONTACT_F solver observables
before stepping and passes those buffers to Newton contact sensors. It does not
implement a second contact-force calculation or modify Newton source.

## Validation and scope

The original configuration completed 4,096-environment training for 1,500 PPO
updates (147,456,000 transitions), seed 42. Final training mean return was 28.76
and mean episode length was 1,000 policy steps (20 seconds). A saved checkpoint
completed a separate 20-second single-environment inference playback with finite
body poses and no early episode reset. These observations establish a working
training/playback configuration, not general robustness or sim-to-real accuracy.

The packaged task passed a five-update 64-environment CLI training check and
a 20-second replay of the demonstrated checkpoint. Its saved body transforms
matched the original playback exactly under the same seed and play settings.
Automated checks also cover the original policy interface, the existing
G1 task's configuration and a physical support-force sensor regression. Conservation, contact accuracy and trajectory
accuracy are not qualified by this walking demonstration. Keep this as an
explicitly selected experimental environment.

To run focused automated checks:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --frozen --extra test python -m pytest \
    source/isaaclab_tasks/test/core/test_velocity_newton_cfg.py
CUDA_VISIBLE_DEVICES=1 uv run --frozen --extra test python -m pytest \
    source/isaaclab_newton/test/physics/test_vbd_core.py
```
