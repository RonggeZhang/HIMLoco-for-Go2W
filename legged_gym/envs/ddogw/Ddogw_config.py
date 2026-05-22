# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class DdogwRoughCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_one_step_observations = 3 + 3 + 3 + 16 + 16 + 16
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 11 * 17 + 12
        num_privileged_obs = num_one_step_privileged_obs * 1
        num_actions = 16

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        static_friction = 0.8
        dynamic_friction = 0.8
        terrain_proportions = [0.1, 0.1, 0.35, 0.2, 0.25]

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 4
        resampling_time = 10.
        heading_command = True

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,

            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 0.8,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,

            'FL_foot_joint': 0.0,
            'RL_foot_joint': 0.0,
            'FR_foot_joint': 0.0,
            'RR_foot_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'hip_joint': 40., 'thigh_joint': 40., 'calf_joint': 40., "foot_joint": 0}
        damping = {'hip_joint': 1, 'thigh_joint': 1, 'calf_joint': 1, "foot_joint": 0.5}
        action_scale = 0.25
        vel_scale = 10.0
        decimation = 4
        wheel_speed = 1

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Ddogw/urdf/Ddogw.urdf'
        name = "Ddogw"
        foot_name = "foot"
        wheel_name = ["foot"]
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ['base_link']
        priviledge_contacts_on = ["thigh", "calf", "base"]
        self_collisions = 1
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.5
            base_height = -10.0
            hip_default = -0.5
            stand_still = -0.5
            collision = -1.0
            feet_stumble = -0.1
            action_rate = -0.01
            torques = -5.0e-4
            dof_vel = -1e-7
            dof_acc = -1e-7
            run_still = -0.05

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.38
        max_contact_force = 100.


class DdogwRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005

    class runner(LeggedRobotCfgPPO.runner):
        save_interval = 1000
        num_steps_per_env = 48
        max_iterations = 20000
        experiment_name = 'Ddogw'
        run_name = ''
        resume = None
        load_run = -1
        checkpoint = -1
        resume_path = None
