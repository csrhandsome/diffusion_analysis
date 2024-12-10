from dataclasses import dataclass
import numpy as np
import mediapy as media
from pathlib import Path
import enum
from tqdm import tqdm
import mujoco
from diffusion.simulation.gripper_env import GripperEnv
import logging
import time

'''def control_gripper(actions, xml_path: str = "data/robotiq_2f85/scene.xml") -> None:
    # 创建 GripperEnv 实例
    env = GripperEnv(xml_path=xml_path)

    # 重置环境到初始状态
    initial_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 初始位姿 [x, y, z, roll, pitch, yaw]
    initial_angle = 0.0  # 初始开度（米）
    env.reset(initial_pose=initial_pose, initial_angle=initial_angle)
    print("环境已重置到初始状态。")

    total_actions = len(actions)
    action_index = 0

    # 初始化仿真参数
    current_time = 0.0
    timestep = env.model.opt.timestep
    max_timestamp = actions[-1]['timestamp'] if total_actions > 0 else 0.0
    # 初始化渲染器
    env.init_render()

    # 开始仿真循环
    while current_time <= max_timestamp or action_index < total_actions:
        # 检查是否有动作需要应用
        while action_index < total_actions and current_time >= actions[action_index]['timestamp']:
            action = actions[action_index]
            pose = action.get('pose')
            angle = action.get('angle')
            print(f"时间 {current_time:.4f} 秒: 应用动作 {action_index + 1}")
            env.step(pose=pose, angle=angle)
            action_index += 1
        # 执行一步模拟，不应用新动作
        env.step()
        # 更新渲染
        if env.is_rendering():
            env.render()
        # 更新当前时间
        current_time += timestep
    
    # 手动关闭渲染器    
    while env.viewer and env.viewer.is_running():
        env.render()
        time.sleep(0.01)
    # 仿真完成，关闭渲染器和环境
    env.close()
    print("仿真已完成并关闭环境。")'''


def control_gripper(env: GripperEnv, pose: np.ndarray, angle: float) -> None:
    # 施加动作
    env.step(pose=pose, angle=angle)
    # 渲染当前状态
    env.render()
    # 延时以观察动作
    time.sleep(env.model.opt.timestep)