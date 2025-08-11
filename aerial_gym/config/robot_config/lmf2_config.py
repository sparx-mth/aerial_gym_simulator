import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig
from aerial_gym.config.sensor_config.lidar_config.pmd_flexx2_config import pmd_flexx2_config




class LMF2Cfg:

    class init_config:
        # init_state tensor is of the format [ratio_x, ratio_y, ratio_z, roll_radians, pitch_radians, yaw_radians, 1.0 (for maintaining shape), vx, vy, vz, wx, wy, wz]
        min_init_state = [
            0.1,
            0.15,
            0.15,
            0,  # -np.pi / 6,
            0,  # -np.pi / 6,
            -np.pi / 6,
            1.0,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
        ]
        max_init_state = [
            0.2,
            0.85,
            0.85,
            0,  # np.pi / 6,
            0,  # np.pi / 6,
            np.pi / 6,
            1.0,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ]

    class sensor_config:
        enable_camera = True
        camera_config = BaseDepthCameraConfig

        enable_lidar = False
        lidar_config = pmd_flexx2_config  # OSDome_64_Config

        enable_imu = False
        imu_config = BaseImuConfig

    class disturbance:
        enable_disturbance = True
        prob_apply_disturbance = 0.05
        max_force_and_torque_disturbance = [4.75, 4.75, 4.75, 0.03, 0.03, 0.03]

    class damping:
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes

    class robot_asset:
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/lmf2"
        file = "model.urdf"
        name = "base_quadrotor"  # actor name
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints.
        fix_base_link = False  # fix the base of the robot
        collision_mask = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        density = 0.000001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.001

        semantic_id = 0
        per_link_semantic = False

        # ------------------------------------ ORIGINAL ---------------------------------------

        # min_state_ratio = [
        #     0.1,
        #     0.1,
        #     0.1,
        #     0,
        #     0,
        #     -np.pi,
        #     1.0,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        # ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        # max_state_ratio = [
        #     0.9,
        #     0.9,
        #     0.9,
        #     0,
        #     0,
        #     np.pi,
        #     1.0,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        # ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]

        # ------------------------------ END ORIGINAL -----------------------------------------

        # ------------------------------ EDIT INITIALIZATION INDOOR ENV ------------------------------
            # center x,y (0.5, 0.5), z at 1.2m -> ratio 0.4 when bounds are 0..3m
        min_state_ratio = [
            0.5, 0.5, 0.4,   # x_ratio, y_ratio, z_ratio
            0.0, 0.0, 0.0,   # roll, pitch, yaw (rad)
            1.0,             # scale
            0.0, 0.0, 0.0,   # vx, vy, vz
            0.0, 0.0, 0.0    # wx, wy, wz
        ]
        max_state_ratio = [
            0.5, 0.5, 0.4,   # same as min => fixed spawn
            0.0, 0.0, 0.0,
            1.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ]
        min_state_ratio[0:2] = [0.47, 0.47]
        max_state_ratio[0:2] = [0.53, 0.53]
        # keep [2] = 0.4 in both

        # --------------------------------------------------------------------------------------------

        max_force_and_torque_disturbance = [
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
        ]  # [fx, fy, fz, tx, ty, tz]

        color = None
        semantic_masked_links = {}
        keep_in_env = True  # this does nothing for the robot

        min_position_ratio = None
        max_position_ratio = None

        min_euler_angles = [-np.pi, -np.pi, -np.pi]
        max_euler_angles = [np.pi, np.pi, np.pi]

        place_force_sensor = True  # set this to True if IMU is desired
        force_sensor_parent_link = "base_link"
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # [x, y, z, qx, qy, qz, qw]

        use_collision_mesh_instead_of_visual = False  # does nothing for the robot

    class control_allocator_config:
        num_motors = 4
        force_application_level = "base_link"  # "motor_link" or "root_link" decides to apply combined forces acting on the robot at the root link or at the individual motor links

        application_mask = [1 + 4 + i for i in range(0, 4)]
        motor_directions = [1, -1, 1, -1]

        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-0.13, -0.13, 0.13, 0.13],
            [-0.13, 0.13, 0.13, -0.13],
            [-0.07, 0.07, -0.07, 0.07],
        ]

        class motor_model_config:
            use_rps = True
            motor_thrust_constant_min = 0.00000926312
            motor_thrust_constant_max = 0.00001826312
            motor_time_constant_increasing_min = 0.05
            motor_time_constant_increasing_max = 0.08
            motor_time_constant_decreasing_min = 0.005
            motor_time_constant_decreasing_max = 0.005
            max_thrust = 10.0
            min_thrust = 0.1
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = 0.07
            use_discrete_approximation = True  # use discrete approximation for motor dynamics
