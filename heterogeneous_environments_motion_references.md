# Configuring Heterogeneous Tasks in Isaac Lab

Isaac Lab's clone-combination feature lets a vectorized task use different physical layouts while retaining one task implementation. The scene owns physical assignment. The task declares which task data belongs to each layout and derives each environment's effective physical signature from `InteractiveScene.clone_plan.clone_mask.T`.

```text
task layout records: InclusionSet + task data
    -> CloneCfg receives the physical InclusionSets
    -> InteractiveScene.clone_plan assigns layouts
    -> the task matches clone_plan.clone_mask.T to its layout records
```

The reference implementation is the file-backed DirectRLEnv maze in `source/isaaclab_tasks/isaaclab_tasks/contrib/heterogeneous_motion/`. Its two layouts have unequal top-level divider counts and select `motions/maze_upper.csv` or `motions/maze_lower.csv`.

> Heterogeneous environment support was introduced in [isaac-sim/IsaacLab#6529](https://github.com/isaac-sim/IsaacLab/pull/6529). This guide builds on the [`heterogeneous_scene.py` demo](scripts/demos/heterogeneous_scene.py#L133-L167) by pairing each physical layout with task-owned motion-reference data. The complete task-only implementation is on the [`vidurv-nvidia/heterogeneous-motion-envs` branch](https://github.com/vidurv-nvidia/IsaacLab/tree/vidurv-nvidia/heterogeneous-motion-envs).

## 1. Choose a construction pattern

Choose the smallest scene representation that matches the physical layouts.

| Pattern | Use when | Example |
| --- | --- | --- |
| Fixed logical slots | Every environment has the same top-level slots, but a slot selects a different asset or compound layout USD. | [`MultiAssetSpawnerCfg`](source/isaaclab_tasks/isaaclab_tasks/contrib/heterogeneous_motion/examples/unequal_assets_scene_cfg.py#L69-L91), [`MultiUsdFileCfg`](source/isaaclab_tasks/isaaclab_tasks/contrib/heterogeneous_motion/examples/fixed_slots_scene_cfg.py#L15-L37) |
| Inclusion sets | Layouts contain different named top-level assets. | [Maze layout definitions](source/isaaclab_tasks/isaaclab_tasks/contrib/heterogeneous_motion/layouts.py#L38-L47), [maze scene fields](source/isaaclab_tasks/isaaclab_tasks/contrib/heterogeneous_motion/scene_cfg.py#L52-L71) |
| `scene.add` | Layouts already exist as separate `InteractiveSceneCfg` objects. | [Separate layouts composed with `scene.add`](source/isaaclab_tasks/isaaclab_tasks/contrib/heterogeneous_motion/examples/unequal_assets_scene_cfg.py#L121-L149) |

The registered manager-based example, `IsaacContrib-Heterogeneous-Motion`, is a legacy procedural construction example. It is not a manager-based maze.

The remaining steps use inclusion sets. For a fixed-slot spawner, keep a parallel task-owned variant assignment: `clone_mask` records that the slot exists, not which asset was selected inside it.

## 2. Define the physical assets

Represent every independently selectable top-level component as a named `InteractiveSceneCfg` field under `{ENV_REGEX_NS}`. Use a kinematic `RigidObjectCfg` for static geometry that must collide under PhysX. Keep global ground and lighting outside the environment namespace.

The maze therefore declares its common agent and perimeter walls plus three optional divider fields. One legal layout selects the upper divider; the other selects the two lower-route dividers.

## 3. Pair each physical layout with its task data

The task owns a small layout record that pairs each physical `InclusionSet` with its motion file. `InclusionSet` itself remains physical-only:

```python
@dataclass(frozen=True)
class LayoutSpec:
    inclusion: InclusionSet
    motion_file: Path


LAYOUTS = (
    LayoutSpec(
        inclusion=InclusionSet(assets=[*_COMMON_ASSETS, "upper_divider"]),
        motion_file=UPPER_MOTION_FILE,
    ),
    LayoutSpec(
        inclusion=InclusionSet(
            assets=[
                *_COMMON_ASSETS,
                "lower_divider_lower",
                "lower_divider_upper",
            ],
        ),
        motion_file=LOWER_MOTION_FILE,
    ),
)
```

## 4. Pass only physical layouts to `CloneCfg`

The scene extracts only the physical part for the clone planner:

```python
clone_cfg = CloneCfg(
    clone_combinations=[layout.inclusion for layout in LAYOUTS]
)
```

## 5. Map physical signatures to layout IDs

After `super().__init__`, group identical rows of `clone_mask.T` and map each physical signature back to its declared `LayoutSpec`:

```python
self.combination_rows, self.combination_ids, self.layout_ids = resolve_layout_ids(self.scene)
self.motion_files = tuple(layout.motion_file for layout in LAYOUTS)
```

`combination_ids` are opaque grouping keys; their numeric order has no task meaning. The task resolves each unique signature by its active asset names, then stores the matching `LAYOUTS` index in `layout_ids`.

## 6. Load and index the task data

Load each file once per declared layout. Keep true lengths and pad a shorter path with its **final sample**, never arbitrary zeros. This keeps terminal goals and nearest-point lookups valid:

```python
self._motion_paths = torch.empty((num_paths, max_length, 2), device=self.device)
for index, path in enumerate(paths):
    self._motion_paths[index] = path[-1]
    self._motion_paths[index, : path.shape[0]] = path

env_paths = self._motion_paths[self.layout_ids]
```

An alternative is to mask all padded samples using the true length. Do not use unmasked zero padding: it can become a false nearest point or goal.
