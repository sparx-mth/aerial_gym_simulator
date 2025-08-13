# Custom SLAM Task + Simple-House Env (Aerial Gym)

This package contains a **ready-to-drop** README, a **ppo YAML** tuned for our task, and a **URDF** for a short internal wall segment. It documents exactly what we changed, how to run it, and GPU 4090 notes.

---

## What’s included (in this bundle)

- `CUSTOM_README.md` (this file) — full instructions
- `ppo_custom_slam.yaml` — rl_games config used in our runs
- `inner_wall_short.urdf` — small wall used to make door gaps

> The code changes below assume your repo layout is similar to:
>
> - `aerial_gym/task/custom_task/custom_slam_task.py` — our task
> - `aerial_gym/config/env_config/simple_house_env_config.py` — new env
> - `aerial_gym/config/asset_config/env_asset_config.py` — add internal walls
> - `resources/models/environment_assets/walls/` — wall URDFs
> - `aerial_gym/config/robot_config/lmf2_config.py` — spawn ratios (LMF2Cfg)

---

## Summary of changes

### 1) Task: `custom_slam_task.py`
Multi‑agent SLAM-ish task (3–5 drones).  
- **Actions (6):** `[vx, vy, vz, yaw_rate, speed_up, slow_down]` → mapped to a 4‑D velocity controller `[vx, vy, vz, yaw_rate]` internally (the last two can act as scalars/biases if desired).  
- **Observations:** fused vector of pose/vel/IMU proxies + coverage stats.  
- **Rewards:** exploration coverage ↑, collision ↓, anti‑fall (discourage negative `vz`), **altitude proxy** (target ≈ 1.5 m via command integration), optional alive bonus.
- **Warmup takeoff:** small upward velocity for first ~100–200 steps so agents don’t drop at t=0.
- **Terminations/Truncations:** time limit truncation + collision-based termination when available.
- **RL‑friendly shapes:** returns `(N,)` rewards/dones for rl_games.

> No direct Isaac Gym calls in the task; if the env later exposes `actor_root_state_tensor`, the task will auto‑use it.

### 2) Env: `simple_house_env_config.py`
12×12×3 m volume, outer walls + interior partitions to create rooms + corridor.  
Add missing flags used by `IGE_env_manager` (e.g. `write_to_sim_at_every_timestep=False`).

Enable internal wall assets via `EnvObjectConfig` mapping:
```python
include_asset_type.update({
    "inner_wall_a": True,
    "inner_wall_b_left": True,
    "inner_wall_b_right": True,
    "inner_wall_c": True,
    "inner_wall_d": True,
})
asset_type_to_dict_map.update({
    "inner_wall_a":      EnvObjectConfig.inner_wall_a,
    "inner_wall_b_left": EnvObjectConfig.inner_wall_b_left,
    "inner_wall_b_right":EnvObjectConfig.inner_wall_b_right,
    "inner_wall_c":      EnvObjectConfig.inner_wall_c,
    "inner_wall_d":      EnvObjectConfig.inner_wall_d,
})
```

### 3) Assets: internal walls in `env_asset_config.py`
Inside `EnvObjectConfig`, add classes (one‑time fixed placements via equal ratios). Example for the **long** segment (you already added this URDF):

```python
class inner_wall_a(base_asset_params):
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
    file = "inner_wall_long.urdf"   # ~6.0 x 0.2 x 3.0 m
    min_state_ratio = [0.5, 0.5, 0.0, 0, 0, 0, 1.0, 0,0,0,0,0,0]
    max_state_ratio = [0.5, 0.5, 0.0, 0, 0, 0, 1.0, 0,0,0,0,0,0]
    collapse_fixed_joints = True
    per_link_semantic = True
    semantic_id = 8
    color = [180, 180, 180]
```

Use the included **`inner_wall_short.urdf`** for `inner_wall_b_left / _right / _d` to create door gaps, and yaw=π/2 where needed for vertical partitions (as in `inner_wall_c`/`d`).

### 4) Robot spawn (LMF2)
In `LMF2Cfg.robot_asset` set fixed spawn at ~1.2 m (z‑ratio ≈ 0.4 for 0..3 m bounds) and centered XY:
```python
min_state_ratio = [0.5, 0.5, 0.4, 0, 0, 0, 1.0, 0,0,0,0,0,0]
max_state_ratio = [0.5, 0.5, 0.4, 0, 0, 0, 1.0, 0,0,0,0,0,0]
```
For multi‑agent spawn, give a small XY spread (e.g., 0.47..0.53) while keeping `z_ratio=0.4`.

---

## How to run

**Visual check (3 envs)**  
```bash
python runner.py \
  --task=custom_slam_task \
  --file=ppo_custom_slam.yaml \
  --train \
  --num_envs=3 \
  --headless=False
```

**Scale up (headless)**  
```bash
python runner.py \
  --task=custom_slam_task \
  --file=ppo_custom_slam.yaml \
  --train \
  --num_envs=256 \
  --headless=True
```

Tip: In the viewer, select the robot actor and “Reset Actor Materials” if it looks too dark. Set the viewer camera to look at z≈1.5 m.

---

## RL config (ppo_custom_slam.yaml)

We include a working rl_games config. Key points:
- Top‑level `params:` preserved so `runner.update_config()` works.
- `num_actors` and `env_config.num_envs` can be overridden via CLI `--num_envs`.

See file: `ppo_custom_slam.yaml` (in this bundle).

---

## GPU 4090 notes

**PyTorch/CUDA**  
Use recent PyTorch with CUDA 12.x for Ada (unless your Isaac Gym build constrains CUDA):
- Conda:
  ```bash
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  ```
- Pip:
  ```bash
  pip install --upgrade pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

**Scaling**  
- On a 4090, try `num_envs: 2048`, `horizon_length: 64`, `minibatch_size: 65536` (must divide batch).
- Keep `mini_epochs: 4`, `e_clip: 0.2`, `gamma: 0.98`, `tau: 0.95`.
- Turn viewer off for throughput.

---

## Troubleshooting

- **Can’t see drones**: confirm spawn z‑ratio, viewer camera, temporarily disable some walls.  
- **NaN/Inf**: `torch.nan_to_num` obs/rewards, clamp obs, maybe lower LR (`5e-5`).  
- **No terminations**: keep time truncate (`episode_len_steps`) and also collision terminate if exposed.  
- **Shape errors**: rewards/dones must be `(N,)` when returning to rl_games.

---

## Optional: real kinematics later

If/when you want true pose/vel from Isaac Gym, expose in `IGE_env_manager`:
```python
from isaacgym import gymtorch
self.actor_root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
# after simulate/fetch each step:
self.gym.refresh_actor_root_state_tensor(self.sim)
```
The task already checks for `sim_env.actor_root_state_tensor` and will use it automatically.

---

## License / attribution

Follows the original repository’s license for all existing code/assets. The added files here are simple configuration/URDF snippets intended for use within that project.

### ✅ Required Fixes Before Training

#### 1. 🧩 Fix Argument Parser in Isaac Gym (`gymutil.py`)

Before training with `aerial_gym_simulator`, modify **line 337** in the `gymutil.py` script of Isaac Gym:

📄 `isaacgym/python/isaacgym/gymutil.py`
```diff
- args = parser.parse_args()
+ args, _ = parser.parse_known_args()
```

🎯 This prevents argument clashes when passing CLI flags for other libraries like `rl_games`.

---

#### 2. 🧩 libstdc++ Fix for CUDA Compatibility (esp. CUDA 12.4 / 4090)

If you encounter errors like:
```
libstdc++.so.6: version `GLIBCXX_3.4.32' not found
```

💡 Install updated C++ runtime from conda-forge:
```bash
conda install -c conda-forge libstdcxx-ng
```

You can also prepend the newer version to your environment:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```