import numpy as np
import mujoco
from typing import List, Tuple

def tune_contact_params(model):
    """调整接触参数以匹配实际力反馈"""
    # 调整接触刚度
    model.opt.elasticity = 1.0
    # 调整阻尼
    model.opt.damping = 0.1
    # 调整摩擦系数
    model.opt.friction = [1.0, 0.005, 0.0001]


def interpolate_actions(current_pose: np.ndarray, target_pose: np.ndarray, steps: int) -> Tuple[List[np.ndarray], List[float]]:
    interpolated_poses = []
    interpolated_angles = []
    for i in range(1, steps + 1):
        alpha = i / steps
        interp_pose = current_pose + alpha * (target_pose - current_pose)
        interpolated_poses.append(interp_pose)
    # 假设开度也是线性变化
    current_angle = env.get_position()
    target_angle = target_pose[-1]  # 或者从另一个参数获取
    interpolated_angles = np.linspace(current_angle, target_angle, steps).tolist()
    return interpolated_poses, interpolated_angles