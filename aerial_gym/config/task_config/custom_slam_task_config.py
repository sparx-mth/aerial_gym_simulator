import torch

class task_config:
    # --- core ---
    seed = 42
    sim_name = "base_sim"
    env_name = "simple_house_env"
    robot_name = "lmf2"
    controller_name = "lmf2_velocity_control"
    num_envs = 16
    agent_count = 3
    use_warp = True
    headless = False
    device = "cuda:0"
    sim_dt = 1.0 / 60.0

    # --- obs / action dims (kept as before) ---
    observation_space_dim = agent_count * (7 + 6 + 6)  # pose + vel + imu (per agent)
    privileged_observation_space_dim = 0
    action_space_dim = 6

    # --- episodes ---
    episode_len_steps = 800
    return_state_before_reset = False

    # --- camera-driven coverage settings ---
    use_camera_coverage = True
    coverage_grid_hw = (512, 512)             # H, W of topdown map (tune vs. speed)
    camera_fov_h_deg = 90.0                 # horizontal FOV (deg)
    camera_fov_v_deg = 0.0                  # 0 = auto infer from image aspect; else set explicitly
    camera_update_every = 2                 # update coverage every k env steps
    cam_stride = 8                          # subsample pixels (bigger=faster)
    cam_max_range_m = 10.0
    cam_min_range_m = 0.2
    ray_hit_epsilon_m = 0.15                # depth “hit” tolerance

    # --- policy warm-up (early bias in action space: adds forward+yaw) ---
    warmup_steps = 60                       # first K steps per env get a bias
    warmup_forward_bias = 0.6               # in [-1,1] before action_transform
    warmup_yaw_sweep = 0.2                  # gentle yaw bias in [-1,1]

    # --- physics warm-up (override controller cmd for takeoff/forward) ---
    physics_warmup_steps = 40               # if >0, forces vz/vx for first K steps
    physics_warmup_vx = 1.0                 # m/s forward
    physics_warmup_vz = 0.5                 # m/s up

    # --- rewards ---
    reward_parameters = {
        # coverage
        "new_area_reward": 10.0,                 # kept for compatibility (not used if you switched to absolute ratio)
        "new_free_reward": 10.0,                 # reward per newly-seen free cell fraction (if you use delta form)
        "new_obstacle_reward": 10.0,             # reward per newly-seen obstacle cell fraction (if you use delta form)
        "coverage_completion_threshold": 0.20,
        "completion_bonus": 500.0,

        # safety / shaping
        "collision_penalty": -5.0,
        "altitude_min_m": 0.8,
        "altitude_max_m": 2.2,
        "altitude_reward_in_range": 1.0,
        "altitude_penalty_out_of_range": -1.0,

        # motion shaping
        "velocity_forward_weight": 0.3,

        # pacing
        "time_penalty": 0.001,
        "stagnation_patience": 40,
    }

    # --- room randomization knobs (consumed by _randomize_room_layout) ---
    # Everything is placed on the floor; unused objects are teleported away.
    args = {
        "room_randomization": {
            "floor_z_m": 0.0,            # z height for furniture
            "wall_margin_m": 0.35,       # keep this distance from walls
            "keep_min": 4,               # min furniture items to keep
            "keep_max": 10,              # max furniture items to keep (capped by available obstacles)
            "radius_min_m": 0.25,        # collision/footprint radius (used for spacing)
            "radius_max_m": 0.45,
            "yaw_min_deg": -180.0,
            "yaw_max_deg":  180.0
        }
    }
