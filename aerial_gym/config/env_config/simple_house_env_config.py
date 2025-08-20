# simple_house_env_config.py
from aerial_gym.config.asset_config.env_asset_config import EnvObjectConfig
import numpy as np

class SimpleHouseEnvCfg(EnvObjectConfig):
    class env:
        # how many parallel envs to spawn
        num_envs = 16
        robots_per_env = 3
        # keep every env layout fixed for the whole run
        keep_same_env_for_num_episodes = 1 
        randomize_rooms_on_reset = True          
        per_env_seed_offset = 1000                # for reproducible per-env randomness

        # number of "environment actions" (optional; leave 0 if not using runtime dynamics yet)
        num_env_actions = 0

        # spacing/bounds (world units in meters)
        env_spacing = 6.0

        # physics/render cadence
        num_physics_steps_per_env_step_mean = 10
        num_physics_steps_per_env_step_std  = 0
        render_viewer_every_n_steps = 1

        # environment behavior
        reset_on_collision = True
        collision_force_threshold = 0.05  # N
        create_ground_plane = False       # we’ll use our own “floor” asset
        sample_timestep_for_latency = True
        perturb_observations = True

        # warp / bounds (12x12x3 m house-like volume)
        use_warp = True
        lower_bound_min = [-6.0, -6.0, 0.0]
        lower_bound_max = [-6.0, -6.0, 0.0]
        upper_bound_min = [ 6.0,  6.0, 3.0]
        upper_bound_max = [ 6.0,  6.0, 3.0]

        write_to_sim_at_every_timestep = False   # no per-step writes (safe default)
        write_to_sim_every_n_steps = 1           # only used if the manager checks this

        experiment_name = "custom_slam_experiment"  # for saving assets, etc.


    class env_config:
        # Turn on outer walls and our new internal walls.
        include_asset_type = {
            # outer shell
            "left_wall": True,
            "right_wall": True,
            "back_wall": True,
            "front_wall": True,
            "top_wall": True,
            "bottom_wall": True,

            # optional “furniture” / obstacles
            "objects": True,
            "panels": False,
            "trees": False,
            "thin": False,

            # internal partitions (we’ll add these classes in EnvObjectConfig)
            "inner_wall_a": True,
            "inner_wall_b_left": True,
            "inner_wall_b_right": True,
            "inner_wall_c": True,
            "inner_wall_d": True,
        }

        # Map names -> classes (must exist in EnvObjectConfig)
        asset_type_to_dict_map = {
            "left_wall":   EnvObjectConfig.left_wall,
            "right_wall":  EnvObjectConfig.right_wall,
            "back_wall":   EnvObjectConfig.back_wall,
            "front_wall":  EnvObjectConfig.front_wall,
            "top_wall":    EnvObjectConfig.top_wall,
            "bottom_wall": EnvObjectConfig.bottom_wall,

            "objects":     EnvObjectConfig.object_asset_params,

            # internal partitions
            "inner_wall_a":      EnvObjectConfig.inner_wall_a,
            "inner_wall_b_left": EnvObjectConfig.inner_wall_b_left,
            "inner_wall_b_right":EnvObjectConfig.inner_wall_b_right,
            "inner_wall_c":      EnvObjectConfig.inner_wall_c,
            "inner_wall_d":      EnvObjectConfig.inner_wall_d,
        }
    
    class viewer:
        headless = False
        ref_env = 0
        camera_position = [-5, -5, 3]
        lookat = [0, 0, 1.5]
        use_collision_geometry = False
        camera_follow_type = "FOLLOW_TRANSFORM"
        camera_follow_position_global_offset = [-2.0, 0.0, 1.0]


