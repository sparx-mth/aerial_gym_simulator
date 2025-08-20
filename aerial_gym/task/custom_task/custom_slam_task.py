from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np
import os
import math
from aerial_gym.utils.custom_metrics import SlamMetrics
from gym.spaces import Box
from isaacgym import gymtorch


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
        self.grid_H, self.grid_W = task_config.coverage_grid_hw
        self.agent_count = task_config.agent_count

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
        # self.action_transformation_function = getattr(self.task_config, "action_transformation_function", self._default_action_tf)

        self.obs_dim = getattr(self.task_config, "observation_space_dim", None)
        if self.obs_dim is None:
            self.obs_dim = 19 * self.agent_count
        self.task_obs = {
            "observations": torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        }

        # Visibility map (3-state)
        self.visibility = torch.zeros((self.num_envs, self.grid_H, self.grid_W), dtype=torch.uint8, device=self.device)
        self.prev_visibility = torch.zeros_like(self.visibility)

        # Rewards
        self.rewards = torch.zeros(self.num_envs, device=self.device)
        self.terminations = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.truncations = torch.zeros_like(self.terminations)

        self.cam_update_every = getattr(self.task_config, "camera_update_every", 2)

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
        
        rp = task_config.reward_parameters

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
        self.reward_new_free = rp.get("new_free_reward", 10.0)
        self.reward_new_obs  = rp.get("new_obstacle_reward", 10.0)
        self.coverage_thresh = rp.get("coverage_completion_threshold", 0.9)
        self.coverage_bonus  = rp.get("completion_bonus", 500.0)

        self.penalty_collision = rp.get("collision_penalty", -5.0)

        self.alt_z_min = rp.get("altitude_min_m", 0.8)
        self.alt_z_max = rp.get("altitude_max_m", 2.2)
        self.alt_reward_in = rp.get("altitude_reward_in_range", 1.0)
        self.alt_penalty_out = rp.get("altitude_penalty_out_of_range", -1.0)

        self.vel_fwd_weight = rp.get("velocity_forward_weight", 0.3)
        self.time_penalty = rp.get("time_penalty", 0.001)
        self.stagnation_patience   = int(rp.get("stagnation_patience", 0))

        # # --- warm-up (single scheme) ---
        # self._warmup_steps = int(getattr(self.task_config, "warmup_steps", 60))  # frames
        # self._fwd_bias_u   = float(getattr(self.task_config, "policy_warmup_forward_bias", 0.6))
        # self._warmup_len   = int(getattr(self.task_config, "warmup_len_steps", 40))  # steps
        # self._warmup_vx    = float(getattr(self.task_config, "warmup_vx_mps", 0.3))
        # self._warmup_vz    = float(getattr(self.task_config, "warmup_vz_mps", 0.8))
        # self._yaw_sweep_u  = float(getattr(self.task_config, "warmup_yaw_sweep", 0.2))     # in [-1,1]

        # --- state for training control ---
        self.global_step = 0
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._no_new_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ground_truth_map = None
        self.root_tensor = None

        # --- metrics ---
        exp_name = getattr(self.task_config, "experiment_name", "custom_slam_experiment")
        log_dir = os.path.join("runs", exp_name)
        self.metrics = SlamMetrics(log_dir=log_dir, num_envs=self.sim_env.num_envs,
                                save_maps_every=200, use_wandb=False)
        # # Metrics
        # self.metrics.log_config({
        #     "reward_new_free": self.reward_new_free,
        #     "coverage_thresh": self.coverage_thresh,
        #     "grid_size": (self.grid_H, self.grid_W)
        # })

     
    # -------------------------------------------------------------------------------------------------------------------------------------
    def step(self, actions):
        self.root_tensor = self.sim_env.IGE_env.vec_root_tensor  # (N_envs, N_assets, 13)
        drone_pos = self.root_tensor[:, 0, 0:3]  # Get the position of all drones
        print("[Debug] vec_root_tensor shape:", self.root_tensor.shape)
        print("[Debug] drone index pos:", self.root_tensor[:, 0, :3])

        self.check_and_reset_envs_with_nan()

        print("[Custom Task] [step] Drone pos:", drone_pos)

        # actions = torch.clamp(torch.tensor(actions, device=self.device), -1, 1)
        actions = torch.clamp(actions.clone().detach(), -1, 1)

        env_actions = self._action_transform(actions)
        self.sim_env.step(env_actions)
        # rewards/terminations (also updates visibility/coverage in your compute)
        rewards, crashes = self.compute_rewards_and_crashes(self.obs_dict)
        coverage_done = (self.visibility > 0).float().mean(dim=(1, 2)) >= self.coverage_thresh
        self.terminations[:] = torch.logical_or(crashes.bool(), coverage_done)
        self.rewards[:] = rewards
        # time-limit truncations
        ep_len = int(getattr(self.task_config, "episode_len_steps", 800))
        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > ep_len,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )
        self.global_step += 1
        self._steps += 1
        self._update_visibility_from_camera()
        seen_now = (self.visibility > 0)
        seen_prev = (self.prev_visibility > 0)
        new_coverage = (seen_now & ~seen_prev).float().mean(dim=(1, 2))
        cov_now = seen_now.float().mean(dim=(1, 2))
        done = cov_now >= self.coverage_thresh
        coverage_done = cov_now >= self.coverage_thresh
        terminations = torch.logical_or(crashes.bool(), coverage_done)
        self.terminations[:] = terminations
        self.rewards[:] = rewards

        # 2. Save binary maps every few steps
        self.metrics.maybe_save_coverage(seen_now, step=self.global_step)

        # Logging coverage metrics
        self.metrics.log_scalar("coverage", cov_now.mean(), self.global_step)
        self.metrics.log_scalar("new_coverage", new_coverage.mean(), self.global_step)

        # Altitude shaping
        alt = self.obs_dict["robot_position"][:, 2]
        alt_reward = torch.where((alt >= self.alt_z_min) & (alt <= self.alt_z_max),
                                 self.alt_reward_in, self.alt_penalty_out)
        self.metrics.log_scalar("altitude", alt.mean(), self.global_step)

        terminated_envs = torch.nonzero(self.terminations).view(-1)
        if len(terminated_envs) > 0:
            print(f"[Custom Task] [Step {self.global_step}] Terminated envs: {terminated_envs.tolist()}")
            for env_id in terminated_envs.tolist():
                reason = []
                if self.terminations[env_id]:
                    reason.append("collision" if self.obs_dict["crashes"][env_id] > 0 else "coverage")
                if self.truncations[env_id]:
                    reason.append("time_limit")
                print(f" - Env {env_id} terminated due to: {', '.join(reason)}")


        num_terminated = self.terminations.sum().item()
        self.metrics.log_scalar("episodes/terminated_envs", num_terminated, self.global_step)

        # if num_terminated > 0:
        #     print(f"[Custom Task] [Step {self.global_step}] Terminated envs: {num_terminated}")
        
        if terminations.any():
            ep_ids = torch.where(terminations)[0]
            ep_returns = rewards[ep_ids]
            ep_lens = self.sim_env.sim_steps[ep_ids]
            ep_covs = cov_now[ep_ids]
            ep_crashes = crashes[ep_ids]

            self.metrics.end_episode(
                env_ids=ep_ids,
                ep_return=ep_returns,
                ep_len=ep_lens,
                final_coverage=ep_covs,
                ep_collisions=ep_crashes,
                coverage_grid=seen_now.float() if seen_now is not None else None
            )



        return self.get_return_tuple()
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def get_return_tuple(self):
        self.process_obs_for_task()
            # This prevents NaNs from being passed to the policy network
        self.task_obs["observations"] = torch.nan_to_num(self.task_obs["observations"], nan=0.0, posinf=0.0, neginf=0.0)

        coverage_done = (self.visibility > 0).float().mean(dim=(1, 2)) >= self.coverage_thresh
        self.terminations[:] = torch.logical_or(self.terminations, coverage_done)
        
        ep_len = int(getattr(self.task_config, "episode_len_steps", 800))
        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > ep_len,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )
        self.global_step += 1
        self._steps += 1
        self._update_visibility_from_camera()
        
        terminated_envs = torch.nonzero(self.terminations).view(-1)
        if len(terminated_envs) > 0:
            print(f"[Custom Task] [Step {self.global_step}] Terminated envs: {terminated_envs.tolist()}")

        return self.task_obs, self.rewards, self.terminations, self.truncations, {}
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def tmp(self, obs_dict):
        a = ['crashes', 'truncations', 'num_env_actions', 'env_actions', 'prev_env_actions', 
             'asset_min_state_ratio', 'asset_max_state_ratio', 'num_obstacles_in_env', 'vec_root_tensor', 
             'robot_state_tensor', 'env_asset_state_tensor', 'unfolded_env_asset_state_tensor', 'unfolded_env_asset_state_tensor_const', 
             'rigid_body_state_tensor', 'global_force_tensor', 'global_torque_tensor', 'unfolded_dof_state_tensor', 'global_contact_force_tensor',
             'robot_contact_force_tensor', 'robot_position', 'robot_orientation', 'robot_linvel', 'robot_angvel', 'robot_body_angvel', 
             'robot_body_linvel', 'robot_euler_angles', 'robot_force_tensor', 'robot_torque_tensor', 'obstacle_position', 'obstacle_orientation', 
             'obstacle_linvel', 'obstacle_angvel', 'obstacle_body_angvel', 'obstacle_body_linvel', 'obstacle_euler_angles', 'obstacle_force_tensor', 
             'obstacle_torque_tensor', 'env_bounds_max', 'env_bounds_min', 'gravity', 'dt', 'CONST_WARP_MESH_ID_LIST', 'CONST_WARP_MESH_PER_ENV', 
             'CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR', 'VERTEX_MAPS_PER_ENV_ORIGINAL', 'robot_mass', 'robot_inertia', 'robot_actions',
             'robot_prev_actions', 'dof_control_mode', 'robot_vehicle_orientation', 'robot_vehicle_linvel', 'num_robot_actions', 'depth_range_pixels', 'segmentation_pixels']

    # -------------------------------------------------------------------------------------------------------------------------------------
    def reset_idx(self, env_ids):
        if self.root_tensor is None:
            self.root_tensor = self.sim_env.IGE_env.vec_root_tensor

        self.sim_env.reset_idx(env_ids)

        # CRITICAL FIX: Manually correct the quaternion to a valid state
        # A valid quaternion for no rotation is (0, 0, 0, 1)
        if self.root_tensor.shape[-1] >= 7:
            # Get the quaternion slice from the actual tensor used by the simulation
            quats = self.obs_dict["vec_root_tensor"][env_ids, 0, 3:7]
            # Replace the invalid quaternion with a valid one for no rotation
            quats[:] = 0.0
            quats[:, 3] = 1.0

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

        # Save ground truth map once
        if self.ground_truth_map is None:
            self.ground_truth_map = self.visibility.clone()
        
        self.infos = {}
        
        # Print the full state of the first drone AFTER the fix for debugging
        print("[Custom Task] [Reset IDX] Full root state AFTER FIX:", self.obs_dict["vec_root_tensor"][env_ids][0, 0])
        print("[Custom Task] [Reset IDX] Drone pos after reset:", self.obs_dict["robot_position"][env_ids])
        
        return
    # -------------------------------------------------------------------------------------------------------------------------------------
    def _action_transform(self, a):
        out = torch.zeros((a.shape[0], 4), device=self.device)
        out[:, 0] = a[:, 0] * 2.0  # vx
        out[:, 2] = a[:, 2] * 2.0  # vz
        out[:, 3] = a[:, 3] * (np.pi / 3)
        return out

    # -------------------------------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_return_tuple()
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def process_obs_for_task(self):
        # use your simplified obs processor: pos, quat, linvel, angvel
        pose, vel, imu = self._extract_state_from_sim()
        flat = torch.cat([pose.view(self.num_envs, -1),
                          vel.view(self.num_envs, -1),
                          imu.view(self.num_envs, -1)], dim=-1)
        flat = flat[:, :self.obs_dim]
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        self.task_obs["observations"].copy_(flat)
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def render(self):
        return self.sim_env.render()

    # -------------------------------------------------------------------------------------------------------------------------------------
    def close(self):
        self.sim_env.delete_env()

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
    def _extract_state_from_sim(self):
        """
        Returns:
        pose: (envs, agents, 7) -> [px,py,pz,qx,qy,qz,qw]
        vel:  (envs, agents, 6) -> [vx,vy,vz, wx,wy,wz]
        imu:  (envs, agents, 6) -> [ax,ay,az, gx,gy,gz] (zeros if not available)
        """
        # Direct and correct access to the root state tensor
        # The previous debug log showed this is the correct path.
        root_states = self.sim_env.IGE_env.vec_root_tensor

        if root_states is None:
            # Fallback: fabricate zeros if the tensor is still not available.
            # This code should ideally not be reached now.
            rm = getattr(self.sim_env, "robot_manager", None)
            if hasattr(rm, "actions") and isinstance(rm.actions, torch.Tensor):
                self.agent_count = int(rm.actions.shape[0])

            pose = torch.zeros((self.num_envs, self.agent_count, 7), device=self.device)
            pose[..., 6] = 1.0  # unit quaternion
            vel = torch.zeros((self.num_envs, self.agent_count, 6), device=self.device)
            imu = torch.zeros((self.num_envs, self.agent_count, 6), device=self.device)
            if not hasattr(self, "_warned_no_states"):
                self._warned_no_states = True
                print("[CustomTask] WARNING: no kinematic state found; using zeroed pose/vel/imu. Wire root states later.")
            return pose, vel, imu

        # Ensure tensor on correct device and normalized to (envs, agents, F)
        root_states = root_states.to(self.device)
        if root_states.dim() == 2:
            F = root_states.shape[-1]
            total = root_states.shape[0]
            if total % self.num_envs == 0:
                inferred_agents = total // self.num_envs
                root_states = root_states.view(self.num_envs, inferred_agents, F)
                self.agent_count = inferred_agents
            else:
                root_states = root_states.unsqueeze(0).repeat(self.num_envs, 1, 1)

        # The usual Isaac root layout: [px,py,pz, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
        F = root_states.shape[-1]
        if F >= 13:
            pos = root_states[..., 0:3]
            quat = root_states[..., 3:7]
            lin_v = root_states[..., 7:10]
            ang_v = root_states[..., 10:13]
        elif F == 12:
            pos = root_states[..., 0:3]
            quat = root_states[..., 3:7]
            lin_v = root_states[..., 7:10]
            ang_v = root_states[..., 10:12]
            pad = torch.zeros((*ang_v.shape[:-1], 1), device=self.device)
            ang_v = torch.cat([ang_v, pad], dim=-1)
        else:
            pos = root_states[..., 0:3]
            quat = torch.zeros((*pos.shape[:-1], 4), device=self.device); quat[..., -1] = 1.0
            lin_v = torch.zeros((*pos.shape[:-1], 3), device=self.device)
            ang_v = torch.zeros((*pos.shape[:-1], 3), device=self.device)

        pose = torch.cat([pos, quat], dim=-1)
        vel = torch.cat([lin_v, ang_v], dim=-1)

        # Handle IMU data (left as is)
        imu = None
        rm = getattr(self.sim_env, "robot_manager", None)
        if hasattr(rm, "imu_buffer"):
            imu_buf = rm.imu_buffer
            imu = imu_buf.view(self.num_envs, self.agent_count, -1).to(self.device)
        elif hasattr(rm, "accel_buffer") and hasattr(rm, "gyro_buffer"):
            acc = rm.accel_buffer.view(self.num_envs, self.agent_count, -1).to(self.device)
            gyro = rm.gyro_buffer.view(self.num_envs, self.agent_count, -1).to(self.device)
            imu = torch.cat([acc, gyro], dim=-1)
        elif hasattr(rm, "get_imu"):
            imu = rm.get_imu()
            if not isinstance(imu, torch.Tensor):
                imu = torch.as_tensor(imu, device=self.device)
            if imu.dim() == 2:
                imu = imu.view(self.num_envs, self.agent_count, -1)

        if imu is None:
            imu = torch.zeros((self.num_envs, self.agent_count, 6), device=self.device)

        return pose, vel, imu
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
        crashes[:] = 0.0 

        # altitude shaping (optional; keep if you want a height band)
        r_alt = 0.0
        if "robot_position" in obs_dict:
            z = obs_dict["robot_position"][:, 2]
            in_range = (z >= self.alt_z_min) & (z <= self.alt_z_max)
            r_alt = torch.where(in_range, self.alt_reward_in, self.alt_penalty_out)

        # final step reward: pay for *new* coverage, not just being alive
        reward = 100.0 * delta_cov + r_alt + self.penalty_collision * crashes  # scale 100 helps signal

        # early “done” if we’ve covered enough
        done_cov = (cov_ratio_now >= self.coverage_thresh)
        reward = reward + done_cov.float() * self.coverage_bonus

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
    def check_and_reset_envs_with_nan(self):
        """
        Scan root tensor for NaN positions and reset affected environments.
        This prevents simulation corruption due to bad actor states.
        """
        # Extract root tensor (B, N, 13) – positions are [:, :, :3]
        drone_positions = self.root_tensor[:, :, :3]

        # Find NaNs (shape: [B, N])
        nan_mask = torch.isnan(drone_positions).any(dim=2)

        # Get env indices where any actor has a NaN
        affected_envs = nan_mask.any(dim=1).nonzero(as_tuple=True)[0]

        if affected_envs.numel() > 0:
            print(f"[NaN Handler] Resetting envs with NaNs: {affected_envs.tolist()}")

            # Optional: print which actor caused the problem
            for env_id in affected_envs:
                for actor_id in range(drone_positions.size(1)):
                    if nan_mask[env_id, actor_id]:
                        print(f" - Env {env_id.item()}, Actor {actor_id}: NaN Detected")

            # Reset the corrupted environments
            self.sim_env.reset_idx(affected_envs)
