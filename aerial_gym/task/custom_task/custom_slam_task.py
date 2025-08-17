from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np
import os
import math


from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.custom_metrics import SlamMetrics
from gym.spaces import Dict, Box
from typing import Dict as TypingDict
from aerial_gym.utils.math import quat_rotate_inverse 


logger = CustomLogger("multiagent_slam_task")


class CustomTask(BaseTask):
    def __init__(self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None, **kwargs):
        # === pass-through overrides (same as NavigationTask) ===
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device

        # === build env exactly like NavigationTask ===
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )
        self.num_envs = self.sim_env.num_envs

        # Pull the in-place observation dict from the env (NavigationTask pattern)
        self.obs_dict = self.sim_env.get_obs()
        if not getattr(self, "_printed_obs_keys", False):
            try:
                ks = list(self.obs_dict.keys())
                print("[CustomTask] obs_dict keys:", ks)
            except Exception:
                pass
            self._printed_obs_keys = True

        # If the env already exposes crashes/truncations, reuse them; else allocate
        self.terminations = self.obs_dict.get(
            "crashes",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        self.truncations = self.obs_dict.get(
            "truncations",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        self.rewards = torch.zeros(self.num_envs, device=self.device)

        # === action / observation spaces ===
        self.action_space = Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_transformation_function = getattr(
            self.task_config, "action_transformation_function", self._default_action_tf
        )

        # If the env exposes single-robot fields like NavigationTask, we can build obs directly.
        # If not, we’ll fall back to your root-state extractor (_extract_state_from_sim()).
        obs_dim = getattr(self.task_config, "observation_space_dim", None)
        if obs_dim is None:
            # default to 19 per agent * agent_count; you already compute this elsewhere if needed
            obs_dim = 19 * getattr(self.task_config, "agent_count", 1)

        self.task_obs = {
            "observations": torch.zeros((self.num_envs, obs_dim), device=self.device, requires_grad=False),
        }

        # --- coverage config ---
        # ---- world bounds (must come before visibility/proj_step_m) ----
        bm = self.obs_dict.get("env_bounds_min", None)
        bx = self.obs_dict.get("env_bounds_max", None)
        self._bounds_min = bm.to(self.device) if isinstance(bm, torch.Tensor) else None  # (N,3) or None
        self._bounds_max = bx.to(self.device) if isinstance(bx, torch.Tensor) else None  # (N,3) or None

        # === 3-state visibility grid (0=unseen, 1=free, 2=obstacle) ===
        self.grid_H, self.grid_W = getattr(self.task_config, "coverage_grid_hw", (100, 100))
        self.visibility = torch.zeros((self.num_envs, self.grid_H, self.grid_W),
                                    dtype=torch.uint8, device=self.device)
        self.prev_visibility = self.visibility.clone()

        # --- camera + projection params ---
        rp = getattr(self.task_config, "reward_parameters", {})
        self.cam_hfov_deg     = float(getattr(self.task_config, "cam_hfov_deg", 90.0))   # horizontal FOV (deg)
        self.cam_max_range_m  = float(getattr(self.task_config, "cam_max_range_m", 10.0))
        self.cam_min_range_m  = float(getattr(self.task_config, "cam_min_range_m", 0.2))
        self.cam_stride       = int(getattr(self.task_config, "cam_stride", 8))          # sample width every N cols
        self.hit_eps_m        = float(getattr(self.task_config, "ray_hit_epsilon_m", 0.15))  # treat near-max as “no hit”

        # step size for ray marching (≈ one grid cell)
        if self._bounds_min is not None and self._bounds_max is not None:
            env_extent = (self._bounds_max[:, :2] - self._bounds_min[:, :2]).mean(dim=0)  # (2,)
            cell_dx = (env_extent[0] / max(1, self.grid_W - 1)).item()
            cell_dy = (env_extent[1] / max(1, self.grid_H - 1)).item()
            self.proj_step_m = max(min(cell_dx, cell_dy), 1e-3)
        else:
            self.proj_step_m = 0.1  # safe default


        # ---- reward weights  ----
        rp = getattr(self.task_config, "reward_parameters", {})
        self.new_free_reward       = float(rp.get("new_free_reward", 10.0))
        self.new_obstacle_reward   = float(rp.get("new_obstacle_reward", 3.0))
        self.cov_completion_thresh = float(rp.get("coverage_completion_threshold", 0.90))
        self.cov_completion_bonus  = float(rp.get("completion_bonus", 500.0))
        self.collision_penalty     = float(rp.get("collision_penalty", -5.0))


        self.coverage = torch.zeros((self.num_envs, self.grid_H, self.grid_W),
                                    dtype=torch.bool, device=self.device)
        self.prev_coverage = torch.zeros_like(self.coverage)
        self.global_step = 0

        # height window reward 
        self.alt_z_min       = float(rp.get("altitude_min_m", 0.8))
        self.alt_z_max       = float(rp.get("altitude_max_m", 2.2))
        self.alt_reward_in   = float(rp.get("altitude_reward_in_range", 1.0))
        self.alt_penalty_out = float(rp.get("altitude_penalty_out_of_range", -1.0))

        self.vel_forward_weight = float(rp.get("velocity_forward_weight", 0.0))
        
        # --- warm-up config & state ---
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._warmup_len = int(getattr(self.task_config, "warmup_len_steps", 40))   # ~0.6s @ 60Hz
        self._warmup_vz  = float(getattr(self.task_config, "warmup_vz_mps", 0.8))   # gentle takeoff
        self._warmup_vx  = float(getattr(self.task_config, "warmup_vx_mps", 0.3))   # small forward nudge
        self.warmup_steps = int(rp.get("warmup_steps", 0))
        self.warmup_vx    = float(rp.get("warmup_vx", 0.0))
        self.warmup_vz    = float(rp.get("warmup_vz", 0.0))

        self.vel_forward_weight = float(rp.get("velocity_forward_weight", 0.0))

        # early truncation (stagnation)
        self.stagnation_patience = int(rp.get("stagnation_patience", 0))
        self._no_new_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.time_penalty = float(rp.get("time_penalty", 0.0))


        # early truncation (stagnation)
        self.stagnation_patience = int(rp.get("stagnation_patience", 0))
        self._no_new_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # small time penalty
        self.time_penalty = float(rp.get("time_penalty", 0.0))



        exp_name = getattr(self.task_config, "experiment_name", "custom_slam_experiment")
        log_dir = os.path.join("runs", exp_name)
        self.metrics = SlamMetrics(log_dir=log_dir, num_envs=self.sim_env.num_envs,
                                   save_maps_every=500, use_wandb=False)  


    def _default_action_tf(self, action: torch.Tensor) -> torch.Tensor:
        # Map 6D -> 4D [vx, vy, vz, yawrate] used by your controller
        # Clamp and scale like NavigationTask does
        a = torch.clamp(action, -1.0, 1.0).to(self.device).float()
        max_speed = 2.0
        max_yawrate = torch.pi / 3
        out = torch.zeros((a.shape[0], 4), device=self.device)
        # Simple mapping: use first three as velocities, 3rd as yaw
        out[:, 0] = a[:, 0] * max_speed
        out[:, 1] = a[:, 1] * 0.0         # lock side-velocity if your controller ignores it
        out[:, 2] = a[:, 2] * max_speed
        out[:, 3] = a[:, 3] * max_yawrate
        return out

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.sim_env.reset_idx(env_ids)
        if isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
            self.coverage[env_ids] = False
            self.prev_coverage[env_ids] = False
            # reset warm-up timer for those envs
            env_ids_dev = env_ids.to(self._steps.device, non_blocking=True)
            self._steps[env_ids_dev] = 0
        self.infos = {}
        return


    def render(self):
        return self.sim_env.render()


    def step(self, actions):
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        env_actions = self.action_transformation_function(actions)  # (N,4): [vx, vy, vz, yawrate]

        # --- warmup: take off + move forward for first K steps per env ---
        if self.warmup_steps > 0:
            mask = (self._steps < self.warmup_steps)  # (N,)
            if mask.any():
                env_actions = env_actions.clone()
                env_actions[mask, 0] = self.warmup_vx    # vx
                env_actions[mask, 1] = 0.0               # vy
                env_actions[mask, 2] = self.warmup_vz    # vz
                env_actions[mask, 3] = 0.0               # yawrate

        self.sim_env.step(actions=env_actions)
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)
        ep_len = getattr(self.task_config, "episode_len_steps", 800)
        self.truncations[:] = torch.where(self.sim_env.sim_steps > ep_len,
                                        torch.ones_like(self.truncations),
                                        torch.zeros_like(self.truncations))
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        # advance step counters AFTER stepping
        self._steps += 1
        # if envs were reset by the sim, clear counters for those
        if len(reset_envs) > 0:
            self._steps[reset_envs] = 0
            self._no_new_counter[reset_envs] = 0        

        # bump global step
        self.global_step += 1

        # save coverage snapshots every few steps
        if hasattr(self, "metrics") and self.metrics is not None:
            try:
                # self.coverage is (N,H,W) bool in your code
                self.metrics.maybe_save_coverage(self.coverage, step=self.global_step)
            except Exception as e:
                if not hasattr(self, "_cov_save_warned"):
                    print(f"[CustomTask] WARNING: failed to save coverage image once: {e}")
                    self._cov_save_warned = True

        # quick textual sanity: log mean coverage every 50 steps
        if (self.global_step % 200) == 0:
            cov_frac = self.coverage.float().mean(dim=(1,2)).mean().item()
            print(f"[CustomTask] step {self.global_step}: mean coverage frac = {cov_frac:.4f}")
        
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)
        ep_len = getattr(self.task_config, "episode_len_steps", 800)
        self.truncations[:] = torch.where(self.sim_env.sim_steps > ep_len,
                                        torch.ones_like(self.truncations),
                                        torch.zeros_like(self.truncations))
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)


        return self.get_return_tuple()

    
    def get_return_tuple(self):
        self.process_obs_for_task()
        return (self.task_obs, self.rewards, self.terminations, self.truncations, self.infos)

    def process_obs_for_task(self):
        # If env provides single-robot fields (NavigationTask style), use them:
        if all(k in self.obs_dict for k in ("robot_position", "robot_vehicle_orientation",
                                            "robot_body_linvel", "robot_body_angvel")):
            # Example packing: [pos(3), quat(4), linvel(3), angvel(3)] = 13 dims
            pos   = self.obs_dict["robot_position"]            # (N,3)
            quat  = self.obs_dict["robot_vehicle_orientation"] # (N,4) or ("robot_orientation")
            lin_v = self.obs_dict["robot_body_linvel"]         # (N,3)
            ang_v = self.obs_dict["robot_body_angvel"]         # (N,3)
            flat = torch.cat([pos, quat, lin_v, ang_v], dim=-1)
            # If your policy expects 19, pad; if 13 is correct for your config, set obs_dim accordingly
            if flat.shape[1] < self.task_obs["observations"].shape[1]:
                pad = torch.zeros((self.num_envs, self.task_obs["observations"].shape[1] - flat.shape[1]), device=self.device)
                flat = torch.cat([flat, pad], dim=-1)
            self.task_obs["observations"].copy_(flat)
            return

        # Otherwise fallback to your multi-agent root-state path
        pose, vel, imu = self._extract_state_from_sim()  # you already have this implemented
        flat_obs = torch.cat(
            [pose.reshape(self.num_envs, -1),
             vel.reshape(self.num_envs, -1),
             imu.reshape(self.num_envs, -1)],
            dim=-1
        )
        # clamp for stability (like you did)
        flat_obs = torch.clamp(flat_obs, -1e3, 1e3)
        self.task_obs["observations"].copy_(flat_obs)

    def compute_rewards_and_crashes(self, obs_dict):
        # update visibility from camera rays
        self._update_visibility_from_camera()

        cur  = self.visibility
        prev = self.prev_visibility

        # env fractions
        newly_free  = ((cur == 1) & (prev != 1)).float().mean(dim=(1, 2))  # (N,)
        newly_obst  = ((cur == 2) & (prev != 2)).float().mean(dim=(1, 2))  # (N,)
        seen_frac   = (cur > 0).float().mean(dim=(1, 2))                   # (N,)
        done_cov    = (seen_frac > self.cov_completion_thresh).float()     # (N,)

        # collisions
        crashes = obs_dict.get("crashes", torch.zeros(self.num_envs, device=self.device))
        crashes = crashes.float().view(-1)

        # altitude reward (as you already had)
        if "robot_position" in obs_dict and getattr(self, "agent_count", 1) == 1:
            z = obs_dict["robot_position"][:, 2].view(self.num_envs, 1)
        else:
            pose, _, _ = self._extract_state_from_sim()
            z = pose[..., 2]  # (N,A)

        in_range = (z >= self.alt_z_min) & (z <= self.alt_z_max)
        r_alt = torch.where(in_range, self.alt_reward_in, self.alt_penalty_out).mean(dim=1)  # (N,)

        # final reward
        reward = (
            self.new_free_reward     * newly_free +
            self.new_obstacle_reward * newly_obst +
            done_cov                 * self.cov_completion_bonus +
            r_alt +
            self.collision_penalty   * crashes
        )

        # lightweight metrics (optional)
        self._last_metrics = {
            "seen_frac":        seen_frac.detach(),
            "newly_free_frac":  newly_free.detach(),
            "newly_obst_frac":  newly_obst.detach(),
            "alt_in_range":     in_range.float().mean(dim=1).detach(),
            "reward":           reward.detach(),
        }
        try:
            if hasattr(self, "metrics") and self.metrics is not None:
                s = getattr(self, "num_task_steps", 0)
                self.metrics.log_scalar("map/seen_frac_mean",        seen_frac.mean().item(), s)
                self.metrics.log_scalar("map/newly_free_frac_mean",  newly_free.mean().item(), s)
                self.metrics.log_scalar("map/newly_obst_frac_mean",  newly_obst.mean().item(), s)
                self.metrics.log_scalar("alt/in_range_frac_mean",    in_range.float().mean().item(), s)
                self.metrics.log_scalar("reward/mean",               reward.mean().item(), s)
        except Exception:
            pass

        if hasattr(self, "metrics") and (getattr(self, "num_task_steps", 0) % 200 == 0):
            try:
                env0_img = self._grid_to_rgb(self.visibility[0]).detach().cpu().numpy()
                self.metrics.save_map_image(env_id=0, img_rgb=env0_img, step=self.num_task_steps)
            except Exception:
                pass


        return reward, crashes


    def _update_visibility_from_camera(self):
        """
        Mark grid cells as seen free (1) along camera rays, and the terminal cell as obstacle (2) if a hit occurs.
        Uses middle scanline of depth_range_pixels as a horizontal fan; falls back to a fixed-range FOV wedge if depth absent.
        """
        self.prev_visibility.copy_(self.visibility)

        # robot XY and yaw per env
        pos = self.obs_dict["robot_position"].to(self.device)
        if pos.dim() == 3:              # (N,A,3) -> mean over agents
            pos_xy = pos[..., :2].mean(dim=1)     # (N,2)
        else:
            pos_xy = pos[:, :2]                   # (N,2)

        if "robot_euler_angles" in self.obs_dict:
            yaw = self.obs_dict["robot_euler_angles"][:, 2].to(self.device)  # (N,)
        else:
            yaw = torch.zeros(self.num_envs, device=self.device)

        # depth sampling (mid row, stride in columns)
        depth = self.obs_dict.get("depth_range_pixels", None)  # expected (N,1,H,W), values in [0..1] scale
        hfov = torch.deg2rad(torch.tensor(self.cam_hfov_deg, device=self.device))

        if depth is not None:
            depth = depth.to(self.device).float()
            H, W = depth.shape[-2], depth.shape[-1]
            row = H // 2
            cols = torch.arange(0, W, step=max(1, self.cam_stride), device=self.device)
            R = cols.numel()

            # meters: clamp >0; treat too-small/invalid as max-range (free to max)
            d = (depth[:, 0, row, cols] * self.cam_max_range_m).clamp(min=0.0)  # (N,R)
            d[d <= self.cam_min_range_m] = self.cam_max_range_m
            thetas = torch.linspace(-hfov / 2, hfov / 2, R, device=self.device)  # (R,)
        else:
            # fallback fan with fixed range
            R = 32
            d = torch.full((self.num_envs, R), self.cam_max_range_m, device=self.device)
            thetas = torch.linspace(-hfov / 2, hfov / 2, R, device=self.device)

        yaw_expanded = yaw.unsqueeze(1) + thetas.unsqueeze(0)  # (N,R)
        dir_xy = torch.stack([torch.cos(yaw_expanded), torch.sin(yaw_expanded)], dim=-1)  # (N,R,2)

        step = self.proj_step_m
        max_steps = int(self.cam_max_range_m / step) + 1

        # small helpers that don't overwrite obstacles with free:
        def _mark_free(n, iy, ix):
            cur = self.visibility[n]
            mask = (cur[iy, ix] == 0)     # only unseen -> free
            if mask.any():
                cur[iy[mask], ix[mask]] = 1

        def _mark_obstacle(n, iy, ix):
            self.visibility[n, iy, ix] = 2  # always write obstacle

        for n in range(self.num_envs):
            p0 = pos_xy[n]  # (2,)
            for r in range(d.shape[1]):
                ray_len = float(d[n, r].item())
                if ray_len <= 0.0:
                    continue

                steps = int(min(max_steps, max(1, math.ceil(ray_len / step))))
                t = torch.linspace(step, steps * step, steps, device=self.device)  # (S,)
                pts = p0.unsqueeze(0) + t.unsqueeze(1) * dir_xy[n, r].unsqueeze(0)  # (S,2)

                # map to grid
                iy, ix = self._xy_to_grid(pts.view(1, -1, 2))  # (1,S)
                iy = iy[0]; ix = ix[0]

                # drop consecutive duplicates (staying inside same cell)
                if iy.numel() > 1:
                    keep = torch.ones_like(iy, dtype=torch.bool)
                    keep[1:] = (iy[1:] != iy[:-1]) | (ix[1:] != ix[:-1])
                    iy = iy[keep]; ix = ix[keep]

                if iy.numel() == 0:
                    continue

                # did the ray hit something? (depth < max_range - eps) -> last cell is obstacle
                hit = ray_len < (self.cam_max_range_m - self.hit_eps_m)
                if hit and iy.numel() >= 1:
                    if iy.numel() > 1:
                        _mark_free(n, iy[:-1], ix[:-1])  # free up to hit
                    _mark_obstacle(n, iy[-1], ix[-1])    # obstacle at hit cell
                else:
                    _mark_free(n, iy, ix)                # no hit: all free
        
    def _xy_to_grid(self, pos_xy: torch.Tensor):
        """Map world XY -> grid indices. pos_xy: (N,A,2). Returns iy, ix as Long (N,A)."""
        H, W = self.grid_H, self.grid_W
        if self._bounds_min is None or self._bounds_max is None:
            min_xy = torch.tensor([-5.0, -5.0], device=self.device).view(1, 1, 2)
            max_xy = torch.tensor([ 5.0,  5.0], device=self.device).view(1, 1, 2)
        else:
            min_xy = self._bounds_min[:, :2].unsqueeze(1)  # (N,1,2)
            max_xy = self._bounds_max[:, :2].unsqueeze(1)  # (N,1,2)

        u = ((pos_xy - min_xy) / (max_xy - min_xy).clamp(min=1e-6)).clamp(0.0, 1.0)
        ix = (u[..., 0] * (W - 1)).round().long().clamp(0, W - 1)
        iy = (u[..., 1] * (H - 1)).round().long().clamp(0, H - 1)
        return iy, ix


    def _update_coverage_from_obs(self):
        """Marks current robot XY cells as visited in self.coverage."""
        self.prev_coverage.copy_(self.coverage)

        # Prefer env-provided positions
        if "robot_position" in self.obs_dict:
            pos = self.obs_dict["robot_position"].to(self.device)
            # Accept (N,3) or (N,A,3)
            if pos.dim() == 2 and pos.shape[0] == self.num_envs and pos.shape[1] == 3:
                pos = pos.view(self.num_envs, 1, 3)             # (N,1,3)
            elif pos.dim() == 3 and pos.shape[-1] == 3:
                pass                                            # (N,A,3)
            else:
                # last-resort reshape if someone changed layout to (N*A,3)
                pos = pos.view(self.num_envs, -1, 3)
        else:
            # Fallback only if absolutely needed
            pose, _, _ = self._extract_state_from_sim()
            pos = pose[..., :3]                                  # (N,A,3)

        iy, ix = self._xy_to_grid(pos[..., :2])                 # (N,A)
        env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(iy)
        self.coverage[env_ids, iy, ix] = True


    def _extract_state_from_sim(self):
        """
        Returns:
        pose: (envs, agents, 7) -> [px,py,pz,qx,qy,qz,qw]
        vel:  (envs, agents, 6) -> [vx,vy,vz, wx,wy,wz]
        imu:  (envs, agents, 6) -> [ax,ay,az, gx,gy,gz] (zeros if not available)
        """
        rm = getattr(self.sim_env, "robot_manager", None)
        if rm is None:
            raise RuntimeError("sim_env has no 'robot_manager' attribute; cannot extract states.")

        # Common Isaac/IsaacGym names to try
        candidate_names = [
            "actor_root_state_tensor",
            "root_state_tensor",
            "root_states",
            "rb_states",
            "rigid_body_states",
            "state_tensor",
            "states",
        ]

        def _try_get_tensor(obj, names):
            for n in names:
                if hasattr(obj, n):
                    t = getattr(obj, n)
                    if isinstance(t, torch.Tensor):
                        return t, n
            return None, None

        # 1) Try on robot_manager
        root_states, src_name = _try_get_tensor(rm, candidate_names)

        # 2) Try directly on sim_env if not found
        if root_states is None:
            root_states, src_name = _try_get_tensor(self.sim_env, candidate_names)

        # 3) Try per-robot list/dict
        if root_states is None:
            robots = getattr(rm, "robots", None)
            if isinstance(robots, (list, tuple)) and len(robots) > 0:
                per = []
                for i, rob in enumerate(robots):
                    t, _ = _try_get_tensor(rob, candidate_names)
                    if t is None:
                        continue
                    per.append(t)
                if per:
                    root_states = torch.stack(per, dim=1)  # (envs, agents, feat) OR (agents, feat)
                    src_name = "robots[*].<state_tensor>"
                    # If shape ends up (agents, feat), expand envs dimension
                    if root_states.dim() == 2:
                        root_states = root_states.unsqueeze(0).repeat(self.num_envs, 1, 1)

        if root_states is None:
            # One-shot debug dump (keep it)
            if not hasattr(self, "_logged_attr_dump"):
                self._logged_attr_dump = True
                print("[CustomTask] Could not find root states. Available robot_manager attrs:")
                for k in dir(rm):
                    if k.startswith("_"):
                        continue
                    try:
                        v = getattr(rm, k)
                        if isinstance(v, torch.Tensor):
                            print(f"  - rm.{k}: shape={tuple(v.shape)}")
                    except Exception:
                        pass
                print("[CustomTask] Available sim_env attrs:")
                for k in dir(self.sim_env):
                    if k.startswith("_"):
                        continue
                    try:
                        v = getattr(self.sim_env, k)
                        if isinstance(v, torch.Tensor):
                            print(f"  - sim_env.{k}: shape={tuple(v.shape)}")
                    except Exception:
                        pass

            # Fallback: fabricate zeros so training can proceed
            # (Optionally infer agent_count from rm.actions)
            if hasattr(rm, "actions") and isinstance(rm.actions, torch.Tensor):
                self.agent_count = int(rm.actions.shape[0])

            pose = torch.zeros((self.num_envs, self.agent_count, 7), device=self.device)
            pose[..., 6] = 1.0  # unit quaternion
            vel  = torch.zeros((self.num_envs, self.agent_count, 6), device=self.device)
            imu  = torch.zeros((self.num_envs, self.agent_count, 6), device=self.device)

            if not hasattr(self, "_warned_no_states"):
                self._warned_no_states = True
                print("[CustomTask] WARNING: no kinematic state found; using zeroed pose/vel/imu. Wire root states later.")

            return pose, vel, imu


        # Ensure tensor on correct device
        if not isinstance(root_states, torch.Tensor):
            root_states = torch.as_tensor(root_states, device=self.device)
        else:
            root_states = root_states.to(self.device)

        # Normalize to (envs, agents, F)
        if root_states.dim() == 2:
            # Expect (envs*agents, F)
            F = root_states.shape[-1]
            total = root_states.shape[0]
            if total % self.num_envs == 0:
                inferred_agents = total // self.num_envs
                root_states = root_states.view(self.num_envs, inferred_agents, F)
                self.agent_count = inferred_agents
            else:
                # fallback: assume agents-first
                root_states = root_states.unsqueeze(0).repeat(self.num_envs, 1, 1)

        # The usual Isaac root layout: [px,py,pz, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
        F = root_states.shape[-1]
        if F >= 13:
            pos   = root_states[..., 0:3]
            quat  = root_states[..., 3:7]
            lin_v = root_states[..., 7:10]
            ang_v = root_states[..., 10:13]
        elif F == 12:
            # Sometimes quat last or missing one comp; make a best-effort split
            pos   = root_states[..., 0:3]
            quat  = root_states[..., 3:7]
            lin_v = root_states[..., 7:10]
            ang_v = root_states[..., 10:12]
            # pad ang_v to 3
            pad = torch.zeros((*ang_v.shape[:-1], 1), device=self.device)
            ang_v = torch.cat([ang_v, pad], dim=-1)
        else:
            # Minimal fallback: positions only; zero the rest
            pos   = root_states[..., 0:3]
            quat  = torch.zeros((*pos.shape[:-1], 4), device=self.device); quat[..., -1] = 1.0
            lin_v = torch.zeros((*pos.shape[:-1], 3), device=self.device)
            ang_v = torch.zeros((*pos.shape[:-1], 3), device=self.device)

        pose = torch.cat([pos, quat], dim=-1)
        vel  = torch.cat([lin_v, ang_v], dim=-1)

        # IMU (optional)
        imu = None
        if hasattr(rm, "imu_buffer"):
            imu_buf = rm.imu_buffer
            imu = imu_buf.view(self.num_envs, self.agent_count, -1).to(self.device)
        elif hasattr(rm, "accel_buffer") and hasattr(rm, "gyro_buffer"):
            acc  = rm.accel_buffer.view(self.num_envs, self.agent_count, -1).to(self.device)
            gyro = rm.gyro_buffer.view(self.num_envs, self.agent_count, -1).to(self.device)
            imu  = torch.cat([acc, gyro], dim=-1)
        elif hasattr(rm, "get_imu"):
            imu = rm.get_imu()
            if not isinstance(imu, torch.Tensor):
                imu = torch.as_tensor(imu, device=self.device)
            if imu.dim() == 2:
                imu = imu.view(self.num_envs, self.agent_count, -1)

        if imu is None:
            imu = torch.zeros((self.num_envs, self.agent_count, 6), device=self.device)

        # One-time info
        if not hasattr(self, "_logged_state_source"):
            self._logged_state_source = True
            print(f"[CustomTask] Using root states from: {src_name}, shape={tuple(root_states.shape)}")

        return pose, vel, imu   


  