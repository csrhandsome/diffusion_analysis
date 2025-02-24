STATE_VEC_IDX_MAPPING = {
    # [0, 6): UR5机械臂关节位置 (6个关节)
    **{
        f'ur5_joint_{i}_pos':i for i in range(6)
    },
    # [6, 12): UR5机械臂关节速度 (6个关节)
    **{
        f'ur5_joint_{i+6}_vel':i for i in range(6)
    },
    # [12, 15): 机械臂末端位置 (x, y, z)
    'eef_pos_x': 12,
    'eef_pos_y': 13,
    'eef_pos_z': 14,
    # [15, 21): 机械臂末端6D姿态 (6个角度)
    'eef_angle_0': 15,
    'eef_angle_1': 16,
    'eef_angle_2': 17,
    'eef_angle_3': 18,
    'eef_angle_4': 19,
    'eef_angle_5': 20,
    # [21, 24): 机械臂末端线速度 (x, y, z)
    'eef_vel_x': 21,
    'eef_vel_y': 22,
    'eef_vel_z': 23,
    # [24, 27): 机械臂末端角速度 (roll, pitch, yaw)
    'eef_angular_vel_roll': 24,
    'eef_angular_vel_pitch': 25,
    'eef_angular_vel_yaw': 26,
    # [27, 28): 夹爪开合角度 (单个值)
    'gripper_angle': 27,
    # [28, 29): 夹爪开合速度 (单个值)
    'gripper_vel': 28,
    # [29, 32): 动捕装置捕捉的末端位置 (x, y, z)
    'mocap_eef_pos_x': 29,
    'mocap_eef_pos_y': 30,
    'mocap_eef_pos_z': 31,
    # [32, 38): 动捕装置捕捉的末端6D姿态 (6个角度)
    'mocap_eef_angle_0': 32,
    'mocap_eef_angle_1': 33,
    'mocap_eef_angle_2': 34,
    'mocap_eef_angle_3': 35,
    'mocap_eef_angle_4': 36,
    'mocap_eef_angle_5': 37,
    # [38, 41): 动捕装置捕捉的末端线速度 (x, y, z)
    'mocap_eef_vel_x': 38,
    'mocap_eef_vel_y': 39,
    'mocap_eef_vel_z': 40,
    # [41, 44): 动捕装置捕捉的末端角速度 (roll, pitch, yaw)
    'mocap_eef_angular_vel_roll': 41,
    'mocap_eef_angular_vel_pitch': 42,
    'mocap_eef_angular_vel_yaw': 43,
    # [44, 64): 预留空间 (20个位置)
    # [64, 128): 预留空间 (64个位置)
}

STATE_VEC_LEN = 128