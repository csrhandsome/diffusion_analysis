import mujoco
import mujoco.viewer  # 确保 Mujoco Viewer 可用
import numpy as np
from typing import Optional, Tuple, List, Dict
import time
from util.pose_transform_util import euler_to_quaternion

class GripperEnv:
    def __init__(self, xml_path: str = "data/stupid_claw/stupid_claw.xml"):
        # 加载 Mujoco 模型
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            print(f"成功加载模型: {xml_path}")
        except Exception as e:
            print(f"无法加载模型: {xml_path}。错误: {e}")
            raise ValueError(f"无法加载模型: {xml_path}。错误: {e}")

        # 创建 Mujoco 数据
        self.data = mujoco.MjData(self.model)
        print("成功初始化 Mujoco 数据。")

        # 获取 actuator 和 site 的 ID
        try:
            # 更新执行器名称为英文，并确保存在于 XML 中
            self.actuator_id_1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotate_joint_1_actuator")
            self.actuator_id_2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotate_joint_2_actuator")
            self.free_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_joint")
        except Exception as e:
            print(f"无法获取 actuator 或 site ID。错误: {e}")
            raise ValueError(f"无法获取 actuator 或 site ID。错误: {e}")

        # 夹爪参数
        self.max_width = 0.1  # 最大开度（米）
        self.min_width = 0.0     # 最小开度（米）
        self.renderer = None
        
        # 存储轨迹点
        self.trajectory = []  
        
        # 控制范围
        self.ctrl_range_1 = self.model.actuator_ctrlrange[self.actuator_id_1] 
        self.ctrl_range_2 = self.model.actuator_ctrlrange[self.actuator_id_2] 

        # 初始化 viewer 为 None
        self.viewer: Optional[mujoco.viewer.MujocoViewer] = None

    def init_render(self) -> None:
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("渲染器已初始化。")

    def step_render(self) -> None:
        if self.viewer is not None and self.viewer.is_running():
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()
            time.sleep(self.model.opt.timestep)

    def step(self, pose: Optional[np.ndarray] = None, angle: Optional[float] = None) -> None:
        if pose is not None:
            self.set_pose(pose)
            print(f"已设置位姿: {pose}")

        if angle is not None:
            self.set_angle(angle)
            print(f"已设置开度: {angle} 米")

        # 执行一步模拟
        mujoco.mj_step(self.model, self.data)
        print(f"执行了一步模拟时间步，时间: {self.model.opt.timestep} 秒。")
        # 在渲染时绘制轨迹

        # 在每一步存储位置
        current_pos,_ = self.get_pose()  # 获取末端执行器位置
        self.trajectory.append(current_pos)
        
        '''# 绘制轨迹
        if self.viewer:
            for i in range(len(self.trajectory)-1):
                self.viewer.add_marker(
                    pos1=self.trajectory[i],
                    pos2=self.trajectory[i+1],
                    size=[0.005],
                    rgba=[0, 1, 0, 0.5],  # 半透明绿色
                    type=mujoco.mjtGeom.mjGEOM_LINE
                )'''

    def set_angle(self, angle: float) -> None:
        # 将开度限制在有效范围内
        width = np.clip(angle, self.min_width, self.max_width)
        print(f"归一化前的宽度: {width} 米")

        # 将开度转换为控制值，根据控制范围
        normalized_width = (width - self.min_width) / (self.max_width - self.min_width)
        ctrl_value = normalized_width * (self.ctrl_range_2[1] - self.ctrl_range_2[0]) + self.ctrl_range_2[0]  # 修改控制范围使用 Joint 2
        print(f"设置的控制值: {ctrl_value}")

        # 设置执行器控制值
        self.data.ctrl[self.actuator_id_1] = ctrl_value 
        self.data.ctrl[self.actuator_id_2] = ctrl_value 

    def get_angle(self) -> float:
        # 从 actuator 的控制值反推开度
        ctrl_value = self.data.ctrl[self.actuator_id]
        normalized_width = (ctrl_value - self.ctrl_range[0]) / (self.ctrl_range[1] - self.ctrl_range[0])
        width = normalized_width * (self.max_width - self.min_width) + self.min_width
        print(f"获取的夹爪开度: {width} 米")
        return width

    def set_pose(self, pose: np.ndarray) -> None:
        if pose.shape != (6,):
            raise ValueError("pose 必须是一个 6 维数组 [x, y, z, roll, pitch, yaw]")

        pos = pose[:3]          # 位置 [x, y, z]
        euler = pose[3:]        # 欧拉角 [roll, pitch, yaw]
        print(f"设置位姿位置: {pos} 米, 欧拉角: {euler} 弧度")
        # 修正高度
        pos[-1]+= 0.6 
        # 将欧拉角转换为四元数
        quat = euler_to_quaternion(euler)
        if quat.shape != (4,):
            raise ValueError("四元数转换失败，应为 4 维数组")
        print(f"转换后的四元数: {quat}")

        # 获取 base_joint 对应的 qpos 索引
        joint_id = self.free_joint_id
        qpos_start_idx = self.model.jnt_qposadr[joint_id]

        # 设置位置
        self.data.qpos[qpos_start_idx:qpos_start_idx+3] = pos
        
        # 设置旋转（四元数）
        self.data.qpos[qpos_start_idx+3:qpos_start_idx+7] = quat

        # 前向运动学更新
        mujoco.mj_forward(self.model, self.data)
        print("已设置新位姿，并向前传播数据。")

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前夹爪位姿。

        Returns:
            Tuple[np.ndarray, np.ndarray]: (位置 [x, y, z], 四元数 [x, y, z, w])。
        """
        # 获取base_joint的qpos起始索引
        qpos_start_idx = self.model.jnt_qposadr[self.free_joint_id]
        
        # 获取位置和四元数
        pos = self.data.qpos[qpos_start_idx:qpos_start_idx+3].copy()
        quat = self.data.qpos[qpos_start_idx+3:qpos_start_idx+7].copy()
        
        print(f"获取的位姿位置: {pos} 米")
        print(f"获取的位姿四元数: {quat}")

        return pos, quat

    def reset(self, initial_pose: Optional[np.ndarray] = None, initial_angle: Optional[float] = None) -> None:
        """
        重置环境到初始状态。

        Args:
            initial_pose (Optional[np.ndarray]): 初始位姿 [x, y, z, roll, pitch, yaw]，形状 (6,)。
            initial_angle (Optional[float]): 初始夹爪开度（米）。
        """
        # 重置 Mujoco 数据
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        print("环境已重置。")

        # 设置初始位姿和开度
        if initial_pose is not None:
            self.set_pose(initial_pose)
            print(f"设置初始位姿: {initial_pose} 米")
        if initial_angle is not None:
            self.set_angle(initial_angle)
            print(f"设置初始夹爪开度: {initial_angle} 米")

        # 执行一步模拟以应用设置
        self.step(pose=initial_pose, angle=initial_angle)
        print("已应用初始设置。")

    def render(self) -> bool:
        # 渲染
        try:
            self.init_render()
            self.step_render()
            return True
        except Exception as e:
            print(f"渲染失败: {e}")
            return False

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            print("关闭渲染器。")

    def get_velocity(self) -> np.ndarray:
        # 假设 qvel[0:3] 是线速度, qvel[3:6] 是角速度
        pos_velocity = self.data.qvel[:3]
        ang_velocity = self.data.qvel[3:6]
        velocity = np.concatenate((pos_velocity, ang_velocity))
        print(f"获取的夹爪速度: {velocity} 米/秒 和 弧度/秒")
        return velocity

    def reset_angle(self, initial_angle: float = 0.0) -> None:
        self.set_angle(initial_angle)
        mujoco.mj_forward(self.model, self.data)
        print(f"已重置夹爪开度到: {initial_angle} 米")

    def is_rendering(self) -> bool:
        rendering = self.viewer is not None and self.viewer.is_running()
        print(f"当前渲染状态: {'渲染中' if rendering else '未渲染'}")
        return rendering