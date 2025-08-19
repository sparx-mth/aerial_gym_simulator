from aerial_gym.env_manager.IGE_env_manager import IsaacGymEnv

from aerial_gym.env_manager.base_env_manager import BaseManager
from aerial_gym.env_manager.asset_manager import AssetManager
from aerial_gym.env_manager.warp_env_manager import WarpEnv
from aerial_gym.env_manager.asset_loader import AssetLoader
from aerial_gym.robots.robot_manager import RobotManagerIGE
from aerial_gym.env_manager.obstacle_manager import ObstacleManager


from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.registry.robot_registry import robot_registry

import torch

from aerial_gym.utils.logging import CustomLogger

import math, random
import numpy as np
from aerial_gym.env_manager.room_layout import RoomParams, sample_room_params, quat_from_yaw


logger = CustomLogger("env_manager")


class EnvManager(BaseManager):
    """
    This class manages the environment. This can handle the creation of the
    robot, environment, and asset managers. This class handles the tensor creation and destruction.

    Moreover, the environment manager can be called within the main environment
    class to manipulate the environment by abstracting the interface.

    This script can remain as generic as possible to handle different types of
    environments, while changes can be made in the individual robot or environment
    managers to handle specific cases.
    """

    def __init__(
        self,
        sim_name,
        env_name,
        robot_name,
        controller_name,
        device,
        args=None,
        num_envs=None,
        use_warp=None,
        headless=None,
    ):
        self.robot_name = robot_name
        self.controller_name = controller_name
        self.sim_config = sim_config_registry.make_sim(sim_name)

        super().__init__(env_config_registry.make_env(env_name), device)

        if num_envs is not None:
            self.cfg.env.num_envs = num_envs
        if use_warp is not None:
            self.cfg.env.use_warp = use_warp
        if headless is not None:
            self.sim_config.viewer.headless = headless

        self.num_envs = self.cfg.env.num_envs
        self.use_warp = self.cfg.env.use_warp

        self.asset_manager = None
        self.tensor_manager = None
        self.env_args = args

        self.keep_in_env = None

        self.global_tensor_dict = {}

        logger.info("Populating environments.")
        self.populate_env(env_cfg=self.cfg, sim_cfg=self.sim_config)
        logger.info("[DONE] Populating environments.")
        self.prepare_sim()

        # store per-env static room layouts (sampled once)
        self._room_params = [None] * self.num_envs
        self._init_room_layouts_once()


        self.sim_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, requires_grad=False, device=self.device
        )

    # -------------------------------------------------------------------------------------------------------------------------------------
    def create_sim(self, env_cfg, sim_cfg):
        """
        This function creates the environment and the robot manager. Does the necessary things to create the environment
        for an IsaacGym environment instance.
        """
        logger.info("Creating simulation instance.")
        logger.info("Instantiating IGE object.")

        # === Need to check this here otherwise IGE will crash with segfault for different CUDA GPUs ====
        has_IGE_cameras = False
        robot_config = robot_registry.get_robot_config(self.robot_name)
        if robot_config.sensor_config.enable_camera == True and self.use_warp == False:
            has_IGE_cameras = True
        # ===============================================================================================

        self.IGE_env = IsaacGymEnv(env_cfg, sim_cfg, has_IGE_cameras, self.device)

        # define a global dictionary to store the simulation objects and important parameters
        # that are shared across the environment, asset, and robot managers
        self.global_sim_dict = {}
        self.global_sim_dict["gym"] = self.IGE_env.gym
        self.global_sim_dict["sim"] = self.IGE_env.sim
        self.global_sim_dict["env_cfg"] = self.cfg
        self.global_sim_dict["use_warp"] = self.IGE_env.cfg.env.use_warp
        self.global_sim_dict["num_envs"] = self.IGE_env.cfg.env.num_envs
        self.global_sim_dict["sim_cfg"] = sim_cfg

        logger.info("IGE object instantiated.")

        if self.cfg.env.use_warp:
            logger.info("Creating warp environment.")
            self.warp_env = WarpEnv(self.global_sim_dict, self.device)
            logger.info("Warp environment created.")

        self.asset_loader = AssetLoader(self.global_sim_dict, self.device)

        logger.info("Creating robot manager.")
        self.robot_manager = RobotManagerIGE(
            self.global_sim_dict, self.robot_name, self.controller_name, self.device
        )
        self.global_sim_dict["robot_config"] = self.robot_manager.cfg
        logger.info("[DONE] Creating robot manager.")

        logger.info("[DONE] Creating simulation instance.")
    
    # -------------------------------------------------------------------------------------------------------------------------------------        
    def _init_room_layouts_once(self):
        """
        Sample and apply a fixed furniture layout per env (once).
        Requires obstacle tensors to be allocated, so call after prepare_sim().
        """
        gd = self.global_tensor_dict
        if ("env_bounds_min" not in gd) or ("env_bounds_max" not in gd):
            logger.warning("[EnvManager] Bounds tensors missing; cannot randomize rooms.")
            return
        bmin = gd["env_bounds_min"][:, :2].detach().cpu().numpy()
        bmax = gd["env_bounds_max"][:, :2].detach().cpu().numpy()

        base_seed = int(getattr(self.cfg.env, "seed", 42))
        seed_stride = int(getattr(self.cfg.env, "per_env_seed_offset", 1000))

        for env_id in range(self.num_envs):
            rng = np.random.RandomState(base_seed + env_id * seed_stride)
            rp = sample_room_params(
                rng=rng,
                bmin_xy=bmin[env_id],
                bmax_xy=bmax[env_id],
                keep_range=(4, 10),         # tweak as you like
                wall_margin=0.35,
                radius_range=(0.25, 0.45),
            )
            self._room_params[env_id] = rp
            self._apply_room_params(env_id, rp)
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def _apply_room_params(self, env_id: int, rp: RoomParams, floor_z: float = 0.0):
        """
        Write the layout into obstacle position/orientation tensors for a single env.
        """
        gd = self.global_tensor_dict
        if ("obstacle_position" not in gd) or ("obstacle_orientation" not in gd):
            logger.warning("[EnvManager] Obstacle tensors missing; cannot apply room layout.")
            return
        pos = gd["obstacle_position"]    # (N, K, 3)
        orn = gd["obstacle_orientation"] # (N, K, 4)
        K = pos.shape[1]
        keep = min(rp.keep, K)

        # place first `keep` items
        for i in range(keep):
            x, y, yaw = rp.furniture[i]["x"], rp.furniture[i]["y"], rp.furniture[i]["yaw"]
            pos[env_id, i, 0] = x
            pos[env_id, i, 1] = y
            pos[env_id, i, 2] = floor_z
            qx, qy, qz, qw = quat_from_yaw(yaw)
            orn[env_id, i, 0] = qx
            orn[env_id, i, 1] = qy
            orn[env_id, i, 2] = qz
            orn[env_id, i, 3] = qw

        # hide extras far away
        for i in range(keep, K):
            pos[env_id, i, 0] = 1e6
            pos[env_id, i, 1] = 1e6
            pos[env_id, i, 2] = 1e6

        # push to sim if available
        ige = getattr(self, "IGE_env", None)
        if ige is not None and hasattr(ige, "write_to_sim"):
            ige.write_to_sim()
    
    # -------------------------------------------------------------------------------------------------------------------------------------    
    def _snapshot_room_layouts(self):
        """
        Creates a static snapshot of the current environment layout tensors.
        This allows resetting or comparing environments without reloading assets.
        """
        t = self.global_tensor_dict

        # Clone unfolded (per-asset, per-env, full state) layout
        unfolded = t.get("unfolded_env_asset_state_tensor", None)
        if isinstance(unfolded, torch.Tensor):
            self._static_layout_unfolded = unfolded.clone()
        else:
            self._static_layout_unfolded = None

        # Clone compact (per-env aggregated) layout
        compact = t.get("env_asset_state_tensor", None)
        if isinstance(compact, torch.Tensor):
            self._static_layout_compact = compact.clone()
        else:
            self._static_layout_compact = None

        # Clone fallback obstacle state (position & orientation only)
        obst_pos = t.get("obstacle_position", None)
        obst_orn = t.get("obstacle_orientation", None)

        self._static_obst_pos = obst_pos.clone() if isinstance(obst_pos, torch.Tensor) else None
        self._static_obst_orn = obst_orn.clone() if isinstance(obst_orn, torch.Tensor) else None

    # -------------------------------------------------------------------------------------------------------------------------------------
    def _reapply_room_layouts(self, env_ids):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        t = self.global_tensor_dict
        if getattr(self, "_static_layout_unfolded", None) is not None and "unfolded_env_asset_state_tensor" in t:
            t["unfolded_env_asset_state_tensor"][env_ids] = self._static_layout_unfolded[env_ids]
            if "unfolded_env_asset_state_tensor_const" in t:
                t["unfolded_env_asset_state_tensor_const"][env_ids] = t["unfolded_env_asset_state_tensor"][env_ids]
        elif getattr(self, "_static_layout_compact", None) is not None and "env_asset_state_tensor" in t:
            t["env_asset_state_tensor"][env_ids] = self._static_layout_compact[env_ids]
        else:
            if "obstacle_position" in t and getattr(self, "_static_obst_pos", None) is not None:
                t["obstacle_position"][env_ids] = self._static_obst_pos[env_ids]
            if "obstacle_orientation" in t and getattr(self, "_static_obst_orn", None) is not None:
                t["obstacle_orientation"][env_ids] = self._static_obst_orn[env_ids]

    # -------------------------------------------------------------------------------------------------------------------------------------
    def populate_env(self, env_cfg, sim_cfg):
        """Populate each environment instance with its assets and robot(s)."""

        # Step 1: Create simulation backend
        self.create_sim(env_cfg, sim_cfg)
        self.robot_manager.create_robot(self.asset_loader)

        # Step 2: Select per-environment assets
        self.global_asset_dicts, keep_in_env_num = self.asset_loader.select_assets_for_sim()

        if self.keep_in_env is None:
            self.keep_in_env = keep_in_env_num
        elif self.keep_in_env != keep_in_env_num:
            raise ValueError(
                f"[populate_env] Inconsistent keep_in_env: expected {self.keep_in_env}, got {keep_in_env_num}"
            )

        # Step 3: Initialize common tensors
        self.step_counter = 0
        self.global_asset_counter = 0
        self.asset_min_state_ratio = []
        self.asset_max_state_ratio = []

        self.global_tensor_dict["crashes"] = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.global_tensor_dict["truncations"] = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.num_env_actions = env_cfg.env.num_env_actions
        self.global_tensor_dict["num_env_actions"] = self.num_env_actions
        self.global_tensor_dict["env_actions"] = None
        self.global_tensor_dict["prev_env_actions"] = None

        self.collision_tensor = self.global_tensor_dict["crashes"]
        self.truncation_tensor = self.global_tensor_dict["truncations"]

        # Optional ground plane
        if env_cfg.env.create_ground_plane:
            logger.info("Creating ground plane...")
            self.IGE_env.create_ground_plane()
            logger.info("Ground plane created.")

        segmentation_ctr = 100

        # Step 4: Populate each individual environment
        for env_id in range(self.num_envs):
            logger.debug(f"Populating environment {env_id}")
            if env_id % 1000 == 0:
                logger.info(f"Progress: Environment {env_id}/{self.num_envs}")

            env_handle = self.IGE_env.create_env(env_id)
            if env_cfg.env.use_warp:
                self.warp_env.create_env(env_id)

            # Step 4a: Add robot(s) to env
            robots_per_env = getattr(env_cfg, "robots_per_env", 1)
            for _ in range(robots_per_env):
                self.robot_manager.add_robot_to_env(
                    self.IGE_env, env_handle, self.global_asset_counter, env_id, segmentation_ctr
                )
                self.global_asset_counter += 1
                segmentation_ctr += 1

            # Step 4b: Add assets to env
            for asset_info in self.global_asset_dicts[env_id]:
                asset_handle, ige_seg = self.IGE_env.add_asset_to_env(
                    asset_info, env_handle, env_id, self.global_asset_counter, segmentation_ctr
                )
                if env_cfg.env.use_warp:
                    _, warp_seg = self.warp_env.add_asset_to_env(
                        asset_info, env_id, self.global_asset_counter, segmentation_ctr
                    )
                else:
                    warp_seg = 0

                segmentation_ctr += max(ige_seg, warp_seg)
                self.global_asset_counter += 1

                # Collect min/max state ratios
                self.asset_min_state_ratio.append(
                    torch.tensor(asset_info["min_state_ratio"], device=self.device)
                )
                self.asset_max_state_ratio.append(
                    torch.tensor(asset_info["max_state_ratio"], device=self.device)
                )

        # Step 5: Stack and store ratio tensors
        if self.asset_min_state_ratio:
            min_ratios = torch.stack(self.asset_min_state_ratio).view(self.num_envs, -1, 13)
            max_ratios = torch.stack(self.asset_max_state_ratio).view(self.num_envs, -1, 13)
        else:
            min_ratios = torch.zeros((self.num_envs, 0, 13), device=self.device)
            max_ratios = torch.zeros((self.num_envs, 0, 13), device=self.device)

        self.global_tensor_dict["asset_min_state_ratio"] = min_ratios
        self.global_tensor_dict["asset_max_state_ratio"] = max_ratios

        self.global_tensor_dict["num_obstacles_in_env"] = len(self.global_asset_dicts[0])
        self._snapshot_room_layouts()


    # -------------------------------------------------------------------------------------------------------------------------------------
    def prepare_sim(self):
        """
        This function prepares the simulation for the environment.
        """
        if not self.IGE_env.prepare_for_simulation(self, self.global_tensor_dict):
            raise Exception("Failed to prepare the simulation")
        if self.cfg.env.use_warp:
            if not self.warp_env.prepare_for_simulation(self.global_tensor_dict):
                raise Exception("Failed to prepare the simulation")
        
        self.asset_manager = AssetManager(self.global_tensor_dict, self.keep_in_env)
        self.asset_manager.prepare_for_sim()
        self.robot_manager.prepare_for_sim(self.global_tensor_dict)
        self.obstacle_manager = ObstacleManager(
            self.IGE_env.num_assets_per_env, self.cfg, self.device
        )
        self.obstacle_manager.prepare_for_sim(self.global_tensor_dict)
        self.num_robot_actions = self.global_tensor_dict["num_robot_actions"]

        self._scatter_floor_objects_once()
        # store per-env static room layouts (sampled once)
        self._room_params = [None] * self.num_envs
        self._init_room_layouts_once()
        self._snapshot_room_layouts()

        # Safe: pick the first available tensor for a quick per-env checksum
        p = None
        for k in ("unfolded_env_asset_state_tensor",
                "env_asset_state_tensor",
                "obstacle_position"):
            v = self.global_tensor_dict.get(k, None)
            if isinstance(v, torch.Tensor) and v.numel() > 0:
                p = v
                break

        if p is not None:
            # sum xyz per env to show layouts are fixed; robust rounding
            dims = tuple(range(1, p.dim()))  # sum over all but env dim
            chk = p[..., :3].sum(dim=dims)
            chk = (chk * 1e4).round() / 1e4
            print("[EnvManager] room checksums per env (fixed):", chk.tolist())

    # -------------------------------------------------------------------------------------------------------------------------------------
    def reset_idx(self, env_ids=None):
        """
        This function resets the environment for the given environment indices.
        """
        # first reset the Isaac Gym environment since that determines the environment bounds
        # then reset the asset managet that respositions assets within the environment
        # then reset the warp environment if it is being used that reads the state tensors from the assets and transforms meshes
        # finally reset the robot manager that resets the robot state tensors and the sensors
        # logger.debug(f"Resetting environments {env_ids}.")
        self.IGE_env.reset_idx(env_ids)
        self.asset_manager.reset_idx(env_ids, self.global_tensor_dict["num_obstacles_in_env"])
        if self.cfg.env.use_warp:
            self.warp_env.reset_idx(env_ids)
        self.robot_manager.reset_idx(env_ids)
        
         # Restore frozen room for those envs
        if hasattr(self, "_room_pos0"):
            s = self.global_tensor_dict["unfolded_env_asset_state_tensor"]
            s[env_ids, :, :3]  = self._room_pos0[env_ids]
            s[env_ids, :, 3:7] = self._room_quat0[env_ids]
            p = self.global_tensor_dict["unfolded_env_asset_state_tensor"][0, :, :3]
            print("[room drift check env0] L2 delta:", float((p - self._room_pos0[0]).abs().max()))
        # Keep rooms static: re-apply the saved layout for these envs
        self._reapply_room_layouts(env_ids)
        self.IGE_env.write_to_sim()
        self.sim_steps[env_ids] = 0

        

    # -------------------------------------------------------------------------------------------------------------------------------------
    def log_memory_use(self):
        """
        This function logs the memory usage of the GPU.
        """
        logger.warning(
            f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/1024/1024/1024}GB"
        )
        logger.warning(
            f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0)/1024/1024/1024}GB"
        )
        logger.warning(
            f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0)/1024/1024/1024}GB"
        )

        # Calculate and system RAM usage used by the objects of this class
        total_memory = 0
        for key, value in self.__dict__.items():
            total_memory += value.__sizeof__()
        logger.warning(
            f"Total memory used by the objects of this class: {total_memory/1024/1024}MB"
        )
    # -------------------------------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.reset_idx(env_ids=torch.arange(self.cfg.env.num_envs))
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def pre_physics_step(self, actions, env_actions):
        # first let the robot compute the actions
        self.robot_manager.pre_physics_step(actions)
        # then the asset manager applies the actions here
        self.asset_manager.pre_physics_step(env_actions)
        # apply actions to obstacle manager
        self.obstacle_manager.pre_physics_step(env_actions)
        # then the simulator applies them here
        self.IGE_env.pre_physics_step(actions)
        # if warp is used, the warp environment applies the actions here
        # If you change the mesh, refit() needs to be called (expensive).
        if self.use_warp:
            self.warp_env.pre_physics_step(actions)

    # -------------------------------------------------------------------------------------------------------------------------------------
    def reset_tensors(self):
        self.collision_tensor[:] = 0
        self.truncation_tensor[:] = 0

    # -------------------------------------------------------------------------------------------------------------------------------------
    def simulate(self, actions, env_actions):
        self.pre_physics_step(actions, env_actions)
        self.IGE_env.physics_step()
        self.post_physics_step(actions, env_actions)

    # -------------------------------------------------------------------------------------------------------------------------------------
    def post_physics_step(self, actions, env_actions):
        self.IGE_env.post_physics_step()
        self.robot_manager.post_physics_step()
        if self.use_warp:
            self.warp_env.post_physics_step()
        self.asset_manager.post_physics_step()

    # -------------------------------------------------------------------------------------------------------------------------------------
    def compute_observations(self):
        self.collision_tensor[:] += (
            torch.norm(self.global_tensor_dict["robot_contact_force_tensor"], dim=1)
            > self.cfg.env.collision_force_threshold
        )

    # -------------------------------------------------------------------------------------------------------------------------------------
    def reset_terminated_and_truncated_envs(self):
        collision_envs = self.collision_tensor.nonzero(as_tuple=False).squeeze(-1)
        truncation_envs = self.truncation_tensor.nonzero(as_tuple=False).squeeze(-1)
        envs_to_reset = (
            (self.collision_tensor * int(self.cfg.env.reset_on_collision) + self.truncation_tensor)
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )
        # reset the environments that have a collision
        if len(envs_to_reset) > 0:
            self.reset_idx(envs_to_reset)
        return envs_to_reset
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def render(self, render_components="sensors"):
        if render_components == "viewer":
            self.render_viewer()
        elif render_components == "sensors":
            self.render_sensors()

    # -------------------------------------------------------------------------------------------------------------------------------------
    def render_sensors(self):
        # render sensors after the physics step
        if self.robot_manager.has_IGE_sensors:
            self.IGE_env.step_graphics()
        self.robot_manager.capture_sensors()

    # -------------------------------------------------------------------------------------------------------------------------------------
    def render_viewer(self):
        # render viewer GUI
        self.IGE_env.render_viewer()

    # -------------------------------------------------------------------------------------------------------------------------------------
    def post_reward_calculation_step(self):
        envs_to_reset = self.reset_terminated_and_truncated_envs()
        # render is performed after reset to ensure that the sensors are updated from the new robot state.
        self.render(render_components="sensors")
        return envs_to_reset
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def step(self, actions, env_actions=None):
        """
        This function steps the simulation for the environment.
        actions: The actions that are sent to the robot.
        env_actions: The actions that are sent to the environment entities.
        """
        self.reset_tensors()
        if env_actions is not None:
            if self.global_tensor_dict["env_actions"] is None:
                self.global_tensor_dict["env_actions"] = env_actions
                self.global_tensor_dict["prev_env_actions"] = env_actions
                self.prev_env_actions = self.global_tensor_dict["prev_env_actions"]
                self.env_actions = self.global_tensor_dict["env_actions"]
            logger.warning(
                f"Env actions shape: {env_actions.shape}, Previous env actions shape: {self.env_actions.shape}"
            )
            self.prev_env_actions[:] = self.env_actions
            self.env_actions[:] = env_actions
        num_physics_step_per_env_step = max(
            math.floor(
                random.gauss(
                    self.cfg.env.num_physics_steps_per_env_step_mean,
                    self.cfg.env.num_physics_steps_per_env_step_std,
                )
            ),
            0,
        )
        for timestep in range(num_physics_step_per_env_step):
            self.simulate(actions, env_actions)
            self.compute_observations()
        self.sim_steps[:] = self.sim_steps[:] + 1
        self.step_counter += 1
        if self.step_counter % self.cfg.env.render_viewer_every_n_steps == 0:
            self.render(render_components="viewer")

    # -------------------------------------------------------------------------------------------------------------------------------------
    def get_obs(self):
        # Just return the dict of all tensors. Whatever the task needs can be used to compute the rewards.
        return self.global_tensor_dict
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    def _scatter_floor_objects_once(self):
        # Random floor placement of existing obstacles (sofa/table/etc. proxies)
        pos = self.global_tensor_dict["obstacle_position"]    # (N,K,3)
        orn = self.global_tensor_dict["obstacle_orientation"] # (N,K,4)
        bmin = self.global_tensor_dict["env_bounds_min"][:, :2]  # (N,2)
        bmax = self.global_tensor_dict["env_bounds_max"][:, :2]  # (N,2)
        N, K = pos.shape[0], pos.shape[1]

        wall_margin = 0.35
        rmin, rmax = 0.25, 0.45
        yaw_lo, yaw_hi = -math.pi, math.pi
        floor_z = 0.0

        for eid in range(N):
            xmin, ymin = (bmin[eid] + wall_margin).tolist()
            xmax, ymax = (bmax[eid] - wall_margin).tolist()
            placed = []
            keep = min(K, torch.randint(4, 10+1, (1,), device=self.device).item())
            tries = 0
            while len(placed) < keep and tries < 2000:
                tries += 1
                rx = float(torch.empty(1, device=self.device).uniform_(xmin, xmax))
                ry = float(torch.empty(1, device=self.device).uniform_(ymin, ymax))
                rr = float(torch.empty(1, device=self.device).uniform_(rmin, rmax))
                if all((rx-px)**2 + (ry-py)**2 >= (rr+pr+0.10)**2 for (px,py,pr) in placed):
                    placed.append((rx, ry, rr))
            # write
            for i in range(K):
                if i < len(placed):
                    rx, ry, _ = placed[i]
                    pos[eid, i, 0] = rx; pos[eid, i, 1] = ry; pos[eid, i, 2] = floor_z
                    yaw = float(torch.empty(1, device=self.device).uniform_(yaw_lo, yaw_hi))
                    cy, sy = math.cos(0.5*yaw), math.sin(0.5*yaw)
                    orn[eid, i, 0] = 0.0; orn[eid, i, 1] = 0.0; orn[eid, i, 2] = sy; orn[eid, i, 3] = cy
                else:
                    pos[eid, i, :].fill_(3e6)  # hide extra objects far away

