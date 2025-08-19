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
        # --- pass-through overrides ---
        if seed is not None:     task_config.seed = seed
        if num_envs is not None: task_config.num_envs = num_envs
        if headless is not None: task_config.headless = headless
        if device is not None:   task_config.device = device
        if use_warp is not None: task_config.use_warp = use_warp
        super().__init__(task_config)
        self.device = self.task_config.device

        # --- env ---
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
        self.agent_count = int(getattr(self.task_config, "agent_count", 1))  # default

        # --- obs dict (in-place tensors from env) ---
        self.obs_dict = self.sim_env.get_obs()
        if not getattr(self, "_printed_obs_keys", False):
            try:
                print("[CustomTask] obs_dict keys:", list(self.obs_dict.keys()))
            except Exception:
                pass
            self._printed_obs_keys = True

        # --- world bounds for XY->grid ---
        self._bounds_min = self.obs_dict.get("env_bounds_min", None)  # (N,3)
        self._bounds_max = self.obs_dict.get("env_bounds_max", None)  # (N,3)

        # --- terminations/truncations/reward vector ---
        self.terminations = self.obs_dict.get("crashes",     torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.truncations  = self.obs_dict.get("truncations", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.rewards      = torch.zeros(self.num_envs, device=self.device)

        # --- action / observation spaces ---
        self.action_space = Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_transformation_function = getattr(self.task_config, "action_transformation_function", self._default_action_tf)

        obs_dim = getattr(self.task_config, "observation_space_dim", None)
        if obs_dim is None:
            obs_dim = 19 * self.agent_count
        self.task_obs = {
            "observations": torch.zeros((self.num_envs, obs_dim), device=self.device, requires_grad=False),
        }

        # --- camera coverage config ---
        rp = getattr(self.task_config, "reward_parameters", {})  # single source of truth
        self.use_camera_coverage = bool(getattr(self.task_config, "use_camera_coverage", True))
        self.grid_H, self.grid_W = getattr(self.task_config, "coverage_grid_hw", (64, 64))

        # FOVs: use explicit H/V if given; else infer vertical FOV from aspect; else square
        self.cam_fov_h_deg = float(getattr(self.task_config, "camera_fov_h_deg", 0.0))
        self.cam_fov_v_deg = float(getattr(self.task_config, "camera_fov_v_deg", 0.0))
        if (self.cam_fov_h_deg <= 0.0) and hasattr(self.obs_dict, "get"):
            # fallback to single "camera_fov_deg" or default 90
            base_fov = float(getattr(self.task_config, "camera_fov_deg", 90.0))
            self.cam_fov_h_deg = base_fov
            depth = self.obs_dict.get("depth_range_pixels", None)
            if isinstance(depth, torch.Tensor) and depth.ndim in (3,4):
                H = depth.shape[-2] if depth.ndim == 3 else depth.shape[-2]
                W = depth.shape[-1] if depth.ndim == 3 else depth.shape[-1]
                # infer vertical from aspect
                self.cam_fov_v_deg = float(2.0 * torch.atan(torch.tan(torch.tensor(self.cam_fov_h_deg * torch.pi/180 / 2)) * (H / max(1.0, float(W)))) * 180.0 / torch.pi)
            else:
                self.cam_fov_v_deg = base_fov
        elif self.cam_fov_v_deg <= 0.0:
            self.cam_fov_v_deg = self.cam_fov_h_deg

        # Near / hit thresholds and ray-march step
        self.cam_min_range_m  = float(rp.get("camera_min_range_m", 0.15))  # ignore depths <= this
        self.hit_eps_m        = float(rp.get("camera_hit_epsilon_m", 0.05))# how close to max range counts as "no hit"

        # If you didn't compute a projection step yet, add a safe default:
        if not hasattr(self, "proj_step_m"):
            self.proj_step_m = 0.1

        self.cam_max_range_m  = float(rp.get("camera_max_range_m", 8.0))
        self.cam_stride       = int(getattr(self.task_config, "camera_stride", 8))          # subsample pixels
        self.cam_update_every = int(getattr(self.task_config, "camera_update_every", 4))    # steps between updates
        self.ray_free_steps   = int(rp.get("ray_free_steps", 12))                           # samples along ray

        # 3-state visibility grid: 0=unseen, 1=seen&free, 2=seen&obstacle
        # ---- visibility & coverage state ----
        # 3-state visibility grid: 0 = unseen, 1 = seen&free, 2 = seen&obstacle
        self.visibility      = torch.zeros((self.num_envs, self.grid_H, self.grid_W),
                                        dtype=torch.uint8, device=self.device)
        self.prev_visibility = torch.zeros_like(self.visibility)

        # Convenience “coverage” bool (any seen) + its previous snapshot
        self.coverage        = torch.zeros((self.num_envs, self.grid_H, self.grid_W),
                                        dtype=torch.bool, device=self.device)
        self.prev_coverage   = torch.zeros_like(self.coverage)

        # layered coverage: 0=unseen, 1=seen&free, 2=seen&obstacle (we keep two bool layers + unseen = inverse)
        self.coverage_seen_free = torch.zeros((self.num_envs, self.grid_H, self.grid_W), dtype=torch.bool, device=self.device)
        self.coverage_seen_obst = torch.zeros_like(self.coverage_seen_free)
        self.coverage_unseen    = torch.ones_like(self.coverage_seen_free)
        self.prev_visibility    = torch.zeros_like(self.coverage_seen_free)  # snapshot of "any seen" last step

        # step size for ray marching (~ one grid cell)
        if (self._bounds_min is not None) and (self._bounds_max is not None):
            env_extent = (self._bounds_max[:, :2] - self._bounds_min[:, :2]).mean(dim=0)  # (2,)
            cell_dx = (env_extent[0] / max(1, self.grid_W - 1)).item()
            cell_dy = (env_extent[1] / max(1, self.grid_H - 1)).item()
            self.proj_step_m = max(min(cell_dx, cell_dy), 1e-3)
        else:
            self.proj_step_m = 0.1

        # --- reward weights ---
        self.new_free_reward       = float(rp.get("new_free_reward", 10.0))
        self.new_obstacle_reward   = float(rp.get("new_obstacle_reward", 3.0))
        self.cov_completion_thresh = float(rp.get("coverage_completion_threshold", 0.90))
        self.cov_completion_bonus  = float(rp.get("completion_bonus", 500.0))
        self.collision_penalty     = float(rp.get("collision_penalty", -5.0))
        self.alt_z_min             = float(rp.get("altitude_min_m", 0.8))
        self.alt_z_max             = float(rp.get("altitude_max_m", 2.2))
        self.alt_reward_in         = float(rp.get("altitude_reward_in_range", 1.0))
        self.alt_penalty_out       = float(rp.get("altitude_penalty_out_of_range", -1.0))
        self.vel_forward_weight    = float(rp.get("velocity_forward_weight", 0.0))
        self.time_penalty          = float(rp.get("time_penalty", 0.0))
        self.stagnation_patience   = int(rp.get("stagnation_patience", 0))

        # --- warm-up (single scheme) ---
        self._warmup_steps = int(getattr(self.task_config, "warmup_steps", 60))  # frames
        self._fwd_bias_u   = float(getattr(self.task_config, "policy_warmup_forward_bias", 0.6))
        self._warmup_len   = int(getattr(self.task_config, "warmup_len_steps", 40))  # steps
        self._warmup_vx    = float(getattr(self.task_config, "warmup_vx_mps", 0.3))
        self._warmup_vz    = float(getattr(self.task_config, "warmup_vz_mps", 0.8))
        self._yaw_sweep_u  = float(getattr(self.task_config, "warmup_yaw_sweep", 0.2))     # in [-1,1]

        self.warmup_steps = int(getattr(self.task_config, "physics_warmup_steps", 0))  # 0 = disabled



        # --- state for training control ---
        self.global_step = 0
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._no_new_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # --- metrics ---
        exp_name = getattr(self.task_config, "experiment_name", "custom_slam_experiment")
        log_dir = os.path.join("runs", exp_name)
        self.metrics = SlamMetrics(log_dir=log_dir, num_envs=self.sim_env.num_envs,
                                save_maps_every=200, use_wandb=False)

 
    # -------------------------------------------------------------------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def close(self):
        self.sim_env.delete_env()

    # -------------------------------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_return_tuple()
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def reset_idx(self, env_ids):
        self.sim_env.reset_idx(env_ids)  # this calls EnvManager.reset_idx above

        if isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
            env_ids = env_ids.to(self.device, non_blocking=True)

            # reset maps/counters only (no layout changes here)
            self.visibility[env_ids] = 0
            self.prev_visibility[env_ids] = 0
            self.coverage_seen_free[env_ids] = False
            self.coverage_seen_obst[env_ids] = False
            self.coverage_unseen[env_ids] = True

            if hasattr(self, "coverage"):       self.coverage[env_ids] = False
            if hasattr(self, "prev_coverage"):  self.prev_coverage[env_ids] = False
            if hasattr(self, "_steps"):         self._steps[env_ids] = 0
            if hasattr(self, "_no_new_counter"): self._no_new_counter[env_ids] = 0
            if isinstance(self.truncations, torch.Tensor):  self.truncations[env_ids] = 0
            if isinstance(self.terminations, torch.Tensor): self.terminations[env_ids] = 0

        self.infos = {}
        return


    # -------------------------------------------------------------------------------------------------------------------------------------
    def render(self):
        return self.sim_env.render()

    # -------------------------------------------------------------------------------------------------------------------------------------
    def step(self, actions):
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)

        # --- Early policy bias: first _warmup_steps frames per env (from sim counter) ---
        warm_k   = int(getattr(self, "_warmup_steps", 0))
        fwd_bias = float(getattr(self, "_fwd_bias_u", 0.0))
        yaw_bias = float(getattr(self, "_yaw_sweep_u", 0.0))

        if warm_k > 0 and hasattr(self.sim_env, "sim_steps"):
            sim_steps = self.sim_env.sim_steps  # (N,)
            # ensure tensor on CPU/GPU agnostic
            if not isinstance(sim_steps, torch.Tensor):
                sim_steps = torch.as_tensor(sim_steps, device=actions.device)
            early = (sim_steps < warm_k).view(-1)
            if early.any():
                a = actions.clone()
                a[early, 0] = torch.clamp(a[early, 0] + fwd_bias, -1.0, 1.0)  # forward nudge
                a[early, 3] = torch.clamp(a[early, 3] + yaw_bias, -1.0, 1.0)  # gentle yaw
                actions = a

        # policy -> controller
        env_actions = self.action_transformation_function(actions)

        # --- Optional physics warmup override for first K steps of each env ---
        phys_k = int(getattr(self, "warmup_steps", 0))
        if phys_k > 0:
            mask = (self._steps < phys_k)  # (N,)
            if mask.any():
                env_actions = env_actions.clone()
                env_actions[mask, 0] = float(getattr(self, "_warmup_vx", 0.3))  # vx
                env_actions[mask, 1] = 0.0                                      # vy
                env_actions[mask, 2] = float(getattr(self, "_warmup_vz", 0.8))  # vz (takeoff)
                env_actions[mask, 3] = 0.0                                      # yawrate

        # step sim
        self.sim_env.step(actions=env_actions)

        # rewards/terminations (also updates visibility/coverage in your compute)
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        # time-limit truncations
        ep_len = int(getattr(self.task_config, "episode_len_steps", 800))
        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > ep_len,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        # post-step housekeeping / possible resets from env
        reset_envs = self.sim_env.post_reward_calculation_step()

        # advance counters AFTER stepping
        self._steps += 1
        self.global_step = getattr(self, "global_step", 0) + 1

        # if some envs were reset by the sim, clear counters/maps
        if isinstance(reset_envs, (list, tuple)):
            reset_envs = torch.as_tensor(reset_envs, device=self.device, dtype=torch.long)
        if isinstance(reset_envs, torch.Tensor) and reset_envs.numel() > 0:
            self.reset_idx(reset_envs)  # your reset_idx zeros _steps (keep that)

        # optional: periodic coverage snapshot
        if hasattr(self, "metrics") and self.metrics is not None:
            try:
                seen_any = (self.coverage_seen_free | self.coverage_seen_obst)
                self.metrics.maybe_save_coverage(seen_any, step=self.global_step)
            except Exception:
                pass

        if (self.global_step % 200) == 0:
            seen_any = (self.coverage_seen_free | self.coverage_seen_obst)
            cov_mean = seen_any.float().mean(dim=(1, 2)).mean().item()
            print(f"[CustomTask] step {self.global_step}: mean seen frac = {cov_mean:.4f}")

        return self.get_return_tuple()

    # -------------------------------------------------------------------------------------------------------------------------------------
    def get_return_tuple(self):
        self.process_obs_for_task()
        return (self.task_obs, self.rewards, self.terminations, self.truncations, self.infos)
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def process_obs_for_task(self):
        obs_dim = self.task_obs["observations"].shape[1]
        # If env provides single-robot fields (NavigationTask style), use them:
        if all(k in self.obs_dict for k in ("robot_position",
                                            "robot_vehicle_orientation",
                                            "robot_body_linvel",
                                            "robot_body_angvel")):
            pos   = self.obs_dict["robot_position"]            # (N,3)
            quat  = self.obs_dict["robot_vehicle_orientation"] # (N,4)
            lin_v = self.obs_dict["robot_body_linvel"]         # (N,3)
            ang_v = self.obs_dict["robot_body_angvel"]         # (N,3)
            flat = torch.cat([pos, quat, lin_v, ang_v], dim=-1)  # (N,13)
        else:
            # Fallback: multi-agent root-states
            pose, vel, imu = self._extract_state_from_sim()      # (N,A,7/6/6)
            flat = torch.cat([
                pose.reshape(self.num_envs, -1),
                vel.reshape(self.num_envs, -1),
                imu.reshape(self.num_envs, -1)
            ], dim=-1)
        # Otherwise fallback to your multi-agent root-state path
        pose, vel, imu = self._extract_state_from_sim()  # you already have this implemented

        flat = flat.to(self.device, non_blocking=True).float()
        if flat.shape[1] < obs_dim:
            pad = torch.zeros((self.num_envs, obs_dim - flat.shape[1]),
                            device=self.device, dtype=flat.dtype)
            flat = torch.cat([flat, pad], dim=-1)
        elif flat.shape[1] > obs_dim:
            flat = flat[:, :obs_dim]

        if not torch.isfinite(flat).all():
            bad = ~torch.isfinite(flat)
            if not hasattr(self, "_printed_bad_obs"):
                envs, cols = bad.nonzero(as_tuple=True)
                print(f"[CustomTask] NON-FINITE OBS: env {envs[:8].tolist()} cols {cols[:8].tolist()}")
                self._printed_bad_obs = True
            flat = torch.nan_to_num(flat, nan=0.0, posinf=1e6, neginf=-1e6)

        flat = torch.clamp(flat, -1e3, 1e3)
        self.task_obs["observations"].copy_(flat)

    # -------------------------------------------------------------------------------------------------------------------------------------
    def compute_rewards_and_crashes(self, obs_dict):
        # update 3-state grid (0 unseen, 1 free, 2 obstacle)
        self._update_visibility_from_camera()

        # seen-any mask
        seen_prev = (self.prev_visibility > 0)
        seen_now  = (self.visibility > 0)
        newly     = seen_now & (~seen_prev)

        # coverage ratios
        cov_ratio_prev = seen_prev.float().mean(dim=(1,2))  # (N,)
        cov_ratio_now  = seen_now.float().mean(dim=(1,2))   # (N,)
        delta_cov      = cov_ratio_now - cov_ratio_prev     # (N,)

        # collisions
        crashes = obs_dict.get("crashes", torch.zeros(self.num_envs, device=self.device)).float().view(-1)

        # altitude shaping (optional; keep if you want a height band)
        r_alt = 0.0
        if "robot_position" in obs_dict:
            z = obs_dict["robot_position"][:, 2]
            in_range = (z >= self.alt_z_min) & (z <= self.alt_z_max)
            r_alt = torch.where(in_range, self.alt_reward_in, self.alt_penalty_out)

        # final step reward: pay for *new* coverage, not just being alive
        reward = 100.0 * delta_cov + r_alt + self.collision_penalty * crashes  # scale 100 helps signal

        # early “done” if we’ve covered enough
        done_cov = (cov_ratio_now >= self.cov_completion_thresh)
        reward = reward + done_cov.float() * self.cov_completion_bonus

        # (optional) time pressure
        if self.time_penalty != 0.0:
            reward = reward - self.time_penalty

        # metrics
        if hasattr(self, "metrics") and self.metrics is not None:
            s = int(getattr(self, "global_step", 0))
            try:
                self.metrics.log_scalar("coverage/ratio_mean", cov_ratio_now.mean().item(), s)
                self.metrics.log_scalar("coverage/delta_mean", delta_cov.mean().item(),  s)
            except Exception:
                pass

        return reward, crashes

    # -------------------------------------------------------------------------------------------------------------------------------------
    def _grid_to_rgb(self, vis_uint8_2d):
        # vis: 0=unseen(black), 1=free(gray), 2=obst(red)
        h, w = vis_uint8_2d.shape
        rgb = torch.zeros((h, w, 3), device=vis_uint8_2d.device, dtype=torch.uint8)
        rgb[vis_uint8_2d == 1] = torch.tensor([180, 180, 180], device=vis_uint8_2d.device, dtype=torch.uint8)
        rgb[vis_uint8_2d == 2] = torch.tensor([220, 50, 50],   device=vis_uint8_2d.device, dtype=torch.uint8)
        return rgb


    # -------------------------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _update_visibility_from_camera(self):
        """
        Mark grid cells as seen free (1) along camera rays, and the terminal cell as obstacle (2) if a hit occurs.
        Uses a horizontal fan from the mid scanline; falls back to a fixed-range wedge if depth is absent.
        """
        if (getattr(self, "global_step", 0) % max(1, self.cam_update_every)) != 0:
            return

        self.prev_visibility.copy_(self.visibility)

        # Pose (N,2) and yaw (N,)
        pos = self.obs_dict["robot_position"].to(self.device)
        pos_xy = (pos[..., :2].mean(dim=1) if pos.dim() == 3 else pos[:, :2])
        yaw = (self.obs_dict["robot_euler_angles"][:, 2].to(self.device)
            if "robot_euler_angles" in self.obs_dict else
            torch.zeros(self.num_envs, device=self.device))

        depth = self.obs_dict.get("depth_range_pixels", None)
        hfov = torch.deg2rad(torch.tensor(self.cam_fov_h_deg, device=self.device))

        if isinstance(depth, torch.Tensor):
            depth = depth.to(self.device).float()
            Himg, Wimg = depth.shape[-2], depth.shape[-1]
            row = Himg // 2
            cols = torch.arange(0, Wimg, step=max(1, self.cam_stride), device=self.device)
            R = cols.numel()

            d = (depth[:, 0, row, cols] * self.cam_max_range_m)  # (N,R)
            d = torch.nan_to_num(d, nan=0.0).clamp(min=0.0)
            d[d <= self.cam_min_range_m] = self.cam_max_range_m

            thetas = torch.linspace(-hfov/2, hfov/2, R, device=self.device)
        else:
            R = 32
            d = torch.full((self.num_envs, R), self.cam_max_range_m, device=self.device)
            thetas = torch.linspace(-hfov/2, hfov/2, R, device=self.device)

        yaw_exp = yaw.unsqueeze(1) + thetas.unsqueeze(0)                 # (N,R)
        dir_xy = torch.stack([torch.cos(yaw_exp), torch.sin(yaw_exp)], dim=-1)  # (N,R,2)

        step = float(self.proj_step_m)
        max_steps = int(self.cam_max_range_m / max(step, 1e-6)) + 1
        H, W = self.grid_H, self.grid_W

        def _mark_free(n, iy, ix):
            cur = self.visibility[n]
            m = (cur[iy, ix] == 0)  # only unseen -> free
            if m.any():
                cur[iy[m], ix[m]] = 1

        def _mark_obstacle(n, iy, ix):
            self.visibility[n, iy, ix] = 2

        # Rays
        for n in range(self.num_envs):
            p0 = pos_xy[n]  # (2,)
            for r in range(d.shape[1]):
                ray_len = float(d[n, r].item())
                if ray_len <= 0.0:
                    continue

                steps = int(min(max_steps, max(1, math.ceil(ray_len / step))))
                t = torch.linspace(step, steps * step, steps, device=self.device)     # (S,)
                pts = p0.unsqueeze(0) + t.unsqueeze(1) * dir_xy[n, r].unsqueeze(0)    # (S,2)

                iy, ix = self._xy_to_grid(pts.view(1, -1, 2))  # (1,S)
                iy = iy[0].reshape(-1); ix = ix[0].reshape(-1)

                # HARD FILTER — prevent any OOB indexing
                valid = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
                if not valid.any():
                    continue
                iy = iy[valid]; ix = ix[valid]

                # drop consecutive duplicates
                if iy.numel() > 1:
                    keep = torch.ones_like(iy, dtype=torch.bool)
                    keep[1:] = (iy[1:] != iy[:-1]) | (ix[1:] != ix[:-1])
                    iy = iy[keep]; ix = ix[keep]
                if iy.numel() == 0:
                    continue

                hit = ray_len < (self.cam_max_range_m - self.hit_eps_m)
                if hit and iy.numel() >= 1:
                    if iy.numel() > 1:
                        _mark_free(n, iy[:-1], ix[:-1])  # free up to the hit cell
                    _mark_obstacle(n, iy[-1], ix[-1])    # obstacle at the hit cell
                else:
                    _mark_free(n, iy, ix)                # no hit → all free

        # Robot cell (also guarded)
        rob_iy, rob_ix = self._xy_to_grid(pos_xy.view(self.num_envs, 1, 2))  # (N,1)
        rob_iy = rob_iy.view(-1); rob_ix = rob_ix.view(-1)
        valid = (rob_iy >= 0) & (rob_iy < H) & (rob_ix >= 0) & (rob_ix < W)
        if valid.any():
            nids = torch.arange(self.num_envs, device=self.device)[valid]
            self.visibility[nids, rob_iy[valid], rob_ix[valid]] = torch.maximum(
                self.visibility[nids, rob_iy[valid], rob_ix[valid]],
                torch.ones_like(rob_iy[valid], dtype=torch.uint8)
            )


    # -------------------------------------------------------------------------------------------------------------------------------------
    def _xy_to_grid(self, pos_xy: torch.Tensor):
        """
        Map world XY -> integer grid indices.
        Accepts (N,K,2) or (N,Hs,Ws,2) or (K,2) or (1,S,2).
        Returns iy, ix with same leading dims as the input (dropping the trailing 2).
        """
        dev = self.device
        H, W = self.grid_H, self.grid_W

        # Normalize input to (N, M, 2) but remember the output shape
        if pos_xy.ndim == 2:                         # (K,2)
            pos_xy = pos_xy.unsqueeze(0)             # (1,K,2)
        if pos_xy.ndim == 3:                         # (N,M,2)
            N, M, _ = pos_xy.shape
            flat = pos_xy
            out_shape = (N, M)
        elif pos_xy.ndim == 4:                       # (N,Hs,Ws,2)
            N, Hs, Ws, _ = pos_xy.shape
            flat = pos_xy.view(N, -1, 2)             # (N,M,2) with M=Hs*Ws
            out_shape = (N, Hs, Ws)
        else:
            raise ValueError(f"_xy_to_grid expects (...,2); got {tuple(pos_xy.shape)}")

        # Per-env bounds
        if (self._bounds_min is None) or (self._bounds_max is None):
            bmin = torch.tensor([-5.0, -5.0], device=dev).view(1, 1, 2).expand(N, flat.shape[1], 2)
            bmax = torch.tensor([ 5.0,  5.0], device=dev).view(1, 1, 2).expand(N, flat.shape[1], 2)
        else:
            bmin2 = self._bounds_min.to(dev)[:, :2]  # (NumEnvs,2)
            bmax2 = self._bounds_max.to(dev)[:, :2]  # (NumEnvs,2)
            if bmin2.shape[0] != N:                  # slice first N if caller passed N=1
                bmin2 = bmin2[:N]; bmax2 = bmax2[:N]
            bmin = bmin2.unsqueeze(1).expand(N, flat.shape[1], 2)
            bmax = bmax2.unsqueeze(1).expand(N, flat.shape[1], 2)

        denom = (bmax - bmin).clamp_min(1e-6)        # avoid /0
        u = ((flat - bmin) / denom).clamp(0.0, 1.0)  # (N,M,2) in [0,1]

        # Use floor and clamp to be extra safe on edges
        ix_flat = torch.clamp((u[..., 0] * (W - 1)).floor().long(), 0, W - 1)
        iy_flat = torch.clamp((u[..., 1] * (H - 1)).floor().long(), 0, H - 1)

        if len(out_shape) == 2:
            return iy_flat, ix_flat                  # (N,M)
        else:
            return iy_flat.view(*out_shape), ix_flat.view(*out_shape)



    # ------------------------------------------------------------------------------------------------------------------------------------- 
    def _yaw_from_quat(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N,4) with convention [qx,qy,qz,qw]. Returns yaw (rad)."""
        qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return torch.atan2(siny_cosp, cosy_cosp)

    # -------------------------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _update_coverage_from_obs(self):
        """Pose-only: mark the robot's current XY cell(s) as visited in self.coverage."""
        # keep previous for "newly covered" computation
        self.prev_coverage.copy_(self.coverage)

        # Get positions from env (preferred), else fall back to root states
        if "robot_position" in self.obs_dict and self.obs_dict["robot_position"] is not None:
            pos = self.obs_dict["robot_position"].to(self.device)  # (N,3) or (N,A,3) or (N*A,3)
            if pos.dim() == 2:                 # (N,3) -> single agent
                pos = pos.view(self.num_envs, 1, 3)
            elif pos.dim() == 3:               # (N,A,3)
                pass
            else:                               # last resort: (N*A,3)
                pos = pos.view(self.num_envs, -1, 3)
        else:
            pose, _, _ = self._extract_state_from_sim()             # (N,A,7)
            pos = pose[..., :3]                                     # (N,A,3)

        # Map XY to grid indices
        pos_xy = pos[..., :2]                                       # (N,A,2)
        iy, ix = self._xy_to_grid(pos_xy)                           # (N,A)

        # Mark cells as seen in the boolean coverage grid
        N, A = iy.shape
        env_ids = torch.arange(N, device=self.device).unsqueeze(1).expand(N, A)
        self.coverage[env_ids, iy, ix] = True

        # If you also keep a 3-state visibility map, optionally mark the robot cell as "seen & free"
        if hasattr(self, "visibility"):
            self.prev_visibility.copy_(self.visibility)
            self.visibility[env_ids, iy, ix] = torch.maximum(
                self.visibility[env_ids, iy, ix],
                torch.ones_like(iy, dtype=torch.uint8)
            )

    # -------------------------------------------------------------------------------------------------------------------------------------
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

    # @torch.no_grad()
    # def _update_coverage_camera(self):
    #     """
    #     Updates self.visibility with a 3-state grid:
    #     0 = unseen, 1 = seen & free, 2 = seen & obstacle
    #     Uses depth + segmentation images; falls back to pose-only if unavailable.
    #     """
    #     # Only work every K steps
    #     step = getattr(self, "global_step", getattr(self, "num_task_steps", 0))
    #     update_every = int(getattr(self, "cam_update_every", 4))
    #     if (step % update_every) != 0:
    #         return

    #     # Require camera tensors; if not present, fallback
    #     if ("depth_range_pixels" not in self.obs_dict) or ("segmentation_pixels" not in self.obs_dict):
    #         self._update_coverage_pose_only()
    #         return

    #     depth = self.obs_dict["depth_range_pixels"]
    #     seg   = self.obs_dict["segmentation_pixels"]
    #     if depth is None or seg is None:
    #         self._update_coverage_pose_only()
    #         return

    #     # (N,1,H,W) -> (N,H,W)
    #     depth = depth.to(self.device).squeeze(1)
    #     seg   = seg.to(self.device).squeeze(1).long()

    #     # Subsample for speed
    #     stride = int(getattr(self, "cam_stride", 8))
    #     d = depth[:, ::stride, ::stride]   # (N,hs,ws)
    #     g =   seg[:, ::stride, ::stride]   # (N,hs,ws)
    #     N, hs, ws = d.shape

    #     # Convert normalized range [0..1] -> meters
    #     cam_max_range = float(getattr(self, "cam_max_range_m", 8.0))
    #     r = d.clamp(0, 1) * cam_max_range                      # (N,hs,ws)

    #     # Robot pose
    #     pos = self.obs_dict["robot_position"].to(self.device)  # (N,3)
    #     yaw = self._yaw_from_quat(self.obs_dict["robot_vehicle_orientation"].to(self.device))  # (N,)

    #     # Intrinsics from FOV (square FOV approx)
    #     fov_deg = float(getattr(self, "cam_fov_deg", getattr(self.task_config, "camera_fov_deg", 90.0)))
    #     fov = torch.tensor(fov_deg, device=self.device) * (torch.pi / 180.0)

    #     xs = torch.linspace(-1, 1, steps=ws, device=self.device) * (fov / 2.0).tan()
    #     ys = torch.linspace(-1, 1, steps=hs, device=self.device) * (fov / 2.0).tan()
    #     grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (hs,ws)

    #     # Ray directions in the body frame (flatten to (1,M))
    #     dir_x = grid_x.reshape(1, -1)
    #     dir_y = grid_y.reshape(1, -1)
    #     norm = torch.sqrt(dir_x * dir_x + dir_y * dir_y + 1.0)
    #     dir_x = dir_x / norm
    #     dir_y = dir_y / norm

    #     # Rotate body->world by yaw (ignore pitch/roll for 2D map)
    #     M    = dir_x.shape[1]
    #     yawN = yaw.view(N, 1)
    #     cosY = torch.cos(yawN); sinY = torch.sin(yawN)
    #     fwd_x =  cosY * dir_x - sinY * dir_y   # (N,M)
    #     fwd_y =  sinY * dir_x + cosY * dir_y   # (N,M)

    #     # Endpoints in world XY
    #     r_flat = r.reshape(N, -1)              # (N,M)
    #     end_x  = pos[:, 0:1] + r_flat * fwd_x  # (N,M)
    #     end_y  = pos[:, 1:2] + r_flat * fwd_y  # (N,M)
    #     end_xy = torch.stack([end_x, end_y], dim=-1).view(N, hs, ws, 2)  # (N,hs,ws,2)

    #     # Save previous visibility before updating
    #     self.prev_visibility.copy_(self.visibility)

    #     # Convert endpoints to grid indices
    #     iy, ix = self._xy_to_grid(end_xy)      # (N,hs,ws)

    #     # env ids for fancy indexing
    #     env_ids = torch.arange(N, device=self.device).view(N, 1, 1).expand(N, hs, ws)

    #     # Free vs obstacle at ray endpoints (adjust class rule if needed)
    #     obst_mask = (g > 0)
    #     free_mask = ~obst_mask

    #     # Obstacles -> 2
    #     self.visibility[env_ids[obst_mask], iy[obst_mask], ix[obst_mask]] = 2

    #     # Free -> max(current, 1)
    #     cur_vals = self.visibility[env_ids[free_mask], iy[free_mask], ix[free_mask]]
    #     self.visibility[env_ids[free_mask], iy[free_mask], ix[free_mask]] = torch.maximum(
    #         cur_vals, torch.ones_like(cur_vals, dtype=torch.uint8)
    #     )

    #     # Mark robot cell as free
    #     rob_iy, rob_ix = self._xy_to_grid(pos[:, :2].view(N, 1, 2))  # (N,1)
    #     self.visibility[torch.arange(N, device=self.device), rob_iy.view(-1), rob_ix.view(-1)] = 1

    #     # Keep your layered coverage in sync (for metrics/snapshots)
    #     self.coverage_seen_free = (self.visibility == 1)
    #     self.coverage_seen_obst = (self.visibility == 2)
    #     self.coverage           = self.coverage_seen_free | self.coverage_seen_obst
    #     self.coverage_unseen    = ~self.coverage


    
    # def _update_coverage_pose_only(self):
    #     self.prev_visibility.copy_(self.visibility)
    #     if "robot_position" in self.obs_dict:
    #         pos = self.obs_dict["robot_position"].to(self.device).view(self.num_envs, 1, 3)
    #     else:
    #         pose, _, _ = self._extract_state_from_sim()
    #         pos = pose[..., :3]
    #     iy, ix = self._xy_to_grid(pos[..., :2])
    #     N, A = iy.shape
    #     env_ids = torch.arange(N, device=self.device).unsqueeze(1).expand(N, A)
    #     self.visibility[env_ids, iy, ix] = torch.maximum(
    #         self.visibility[env_ids, iy, ix],
    #         torch.ones_like(ix, dtype=torch.uint8)
    #     )
           

    # # -------------------------------------------------------------------------------------------------------------------------------------
    # def _randomize_room_layout(self, env_ids: torch.Tensor):
    #     """Reposition obstacle actors onto the floor with non-overlapping footprints."""
    #     if not isinstance(env_ids, torch.Tensor):
    #         env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    #     if env_ids.numel() == 0:
    #         return

    #     # bounds in XY
    #     bmin = self.obs_dict["env_bounds_min"][:, :2]  # (N,2)
    #     bmax = self.obs_dict["env_bounds_max"][:, :2]  # (N,2)

    #     # tunables (or pull from task_config.args["room_randomization"])
    #     wall_margin   = 0.35
    #     floor_z       = float(self.task_config.args.get("room_randomization", {}).get("floor_z_m", 0.0))
    #     rmin, rmax    = (0.25, 0.45)   # footprint radii (m)
    #     yaw_lo, yaw_hi = (-math.pi, math.pi)
    #     # how many items to keep per env (cap by available K)
    #     keep_lo, keep_hi = (4, 10)

    #     pos = self.obs_dict["obstacle_position"]    # (N, K, 3)
    #     orn = self.obs_dict["obstacle_orientation"] # (N, K, 4)
    #     K = pos.shape[1]

    #     for eid in env_ids.tolist():
    #         xmin, ymin = bmin[eid]
    #         xmax, ymax = bmax[eid]
    #         xmin += wall_margin; ymin += wall_margin
    #         xmax -= wall_margin; ymax -= wall_margin

    #         keep = int(torch.randint(keep_lo, min(keep_hi, K)+1, (1,), device=self.device).item())
    #         placed_xy, radii = [], []
    #         trials = 0
    #         while len(placed_xy) < keep and trials < 2000:
    #             trials += 1
    #             rx = float(torch.empty(1, device=self.device).uniform_(xmin, xmax).item())
    #             ry = float(torch.empty(1, device=self.device).uniform_(ymin, ymax).item())
    #             rr = float(torch.empty(1, device=self.device).uniform_(rmin, rmax).item())

    #             ok = True
    #             for (px, py), pr in zip(placed_xy, radii):
    #                 if (rx - px)**2 + (ry - py)**2 < (rr + pr + 0.10)**2:
    #                     ok = False; break
    #             if ok:
    #                 placed_xy.append((rx, ry)); radii.append(rr)

    #         # write first `keep` items; hide the rest far away
    #         for i in range(K):
    #             if i < len(placed_xy):
    #                 rx, ry = placed_xy[i]
    #                 pos[eid, i, 0] = rx
    #                 pos[eid, i, 1] = ry
    #                 pos[eid, i, 2] = floor_z
    #                 yaw = float(torch.empty(1, device=self.device).uniform_(yaw_lo, yaw_hi).item())
    #                 cy, sy = math.cos(0.5*yaw), math.sin(0.5*yaw)
    #                 orn[eid, i, 0] = 0.0
    #                 orn[eid, i, 1] = 0.0
    #                 orn[eid, i, 2] = sy
    #                 orn[eid, i, 3] = cy
    #             else:
    #                 pos[eid, i, 0] = 1e6
    #                 pos[eid, i, 1] = 1e6
    #                 pos[eid, i, 2] = 1e6

    #     # push to sim if your IGE manager exposes a write hook
    #     ige = getattr(self.sim_env, "IGE_env", None)
    #     if ige is not None and hasattr(ige, "write_to_sim"):
    #         ige.write_to_sim()




    