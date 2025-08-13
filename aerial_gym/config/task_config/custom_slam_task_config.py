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
        "exploration_weight": 10.0,
        "collision_penalty": -5.0,
        "altitude_reward_in_range": 1.0,
        "altitude_penalty_out_of_range": -1.0,
        "anti_fall_up_bonus": 0.5,
        "anti_fall_down_penalty": -1.0,
        "velocity_forward_weight": 0.5,
        "new_area_reward": 10.0,
        "coverage_completion_threshold": 0.90,  # 90% of space
        "completion_bonus": 500.0
        }
    sim_dt = 1.0/60.0                     # for your altitude proxy integrator
