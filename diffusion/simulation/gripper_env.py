import mujoco
import mujoco.viewer
import numpy as np
import time

class Mujoco_GripperEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("diffusion/simulation/structure.xml")
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.trajectory = []
    
    def set_pose(self, pose):
        """设置夹爪位姿"""
        pose = np.array(pose, dtype=np.float64)
        self.data.qpos[:6] = pose
        mujoco.mj_forward(self.model, self.data)
    
    def set_gripper_angle(self, angle):
        """设置夹爪开合角度"""
        angle_array = np.array([angle/2, -angle/2], dtype=np.float64)
        self.data.qpos[6] = angle_array[0]
        self.data.qpos[7] = angle_array[1]
        mujoco.mj_forward(self.model, self.data)
    
    def get_forces(self):
        """获取接触力"""
        left_force = self.data.sensor('left_force').data
        right_force = self.data.sensor('right_force').data
        return left_force, right_force
    
    def step(self):
        """执行一步仿真"""
        mujoco.mj_step(self.model, self.data)
    
    def render(self):
        """渲染当前场景"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        if self.viewer.is_running():
            self.viewer.sync()
            return True
        return False
    
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def add_trajectory_point(self):
        """添加轨迹点"""
        pos = self.data.qpos[:3].copy()
        self.trajectory.append(pos)
    
    def reset(self):
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        self.trajectory = []