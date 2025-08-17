import torch


class task_config:
    seed = 42
    sim_name = "base_sim"
    env_name = "simple_house_env"         
    robot_name = "lmf2"
    controller_name = "lmf2_velocity_control"
    args = {}
    num_envs = 16
    agent_count = 3
    use_warp = True
    headless = False
    device = "cuda:0"

    observation_space_dim = agent_count * (7 + 6 + 6)  # pose + velocity + IMU
    privileged_observation_space_dim = 0
    action_space_dim = 6

    episode_len_steps = 800
    return_state_before_reset = False
    reward_parameters = {
            "new_area_reward": 10.0,
            "collision_penalty": -5.0,
            "coverage_completion_threshold": 0.90,
            "completion_bonus": 500.0,
            "altitude_min_m": 0.8,                 # start of “good” height band
            "altitude_max_m": 2.2,                 # end of “good” height band
            "altitude_reward_in_range": 1.0,       # reward per env when agents are in band
            "altitude_penalty_out_of_range": -1.0,  # penalty per env when agents are out of band
            "velocity_forward_weight": 0.3,     # >0 to encourage forward motion
            "warmup_steps": 60,                 # ~1 s at 60 Hz
            "warmup_vx": 1.0,                   # m/s forward
            "warmup_vz": 0.5,                   # m/s up
            "time_penalty": 0.001,              # small per-step penalty
            "stagnation_patience": 40,           # early-truncate if no new cells this many steps
            "new_free_reward": 10.0,
            "new_obstacle_reward": 10.0       
        }

    sim_dt = 1.0/60.0                     # for your altitude proxy integrator
    cam_hfov_deg: 90.0
    cam_max_range_m: 10.0
    cam_min_range_m: 0.2
    cam_stride: 8
    ray_hit_epsilon_m: 0.15
