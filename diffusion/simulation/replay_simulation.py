from diffusion.simulation.gripper_env import Mujoco_GripperEnv
import time

class SimpleSimulator:
    def __init__(self, dt=0.002):
        self.env = Mujoco_GripperEnv()
        self.dt = dt
    
    def step(self, pose, angle):
        """执行单步仿真"""
        try:
            # 设置位姿和角度
            self.env.set_pose(pose)
            self.env.set_gripper_angle(angle)
            
            # 执行仿真步进
            self.env.step()
            
            # 渲染场景
            self.env.render()
            
            # 获取力数据（如果需要）
            left_force, right_force = self.env.get_forces()
            
            # 控制仿真速度
            time.sleep(self.dt)
            
            return left_force, right_force
            
        except Exception as e:
            print(f"Simulation error: {e}")
            return None
    
    def close(self):
        """关闭环境"""
        self.env.close()

# 使用示例：
def simple_simulation(pose, angle):
    simulator = SimpleSimulator()
    try:
        forces = simulator.step(pose, angle)
        return forces
    finally:
        simulator.close()

# 如果需要连续多次调用但不想每次都创建新环境：
def create_simulator():
    return SimpleSimulator()

# 使用示例
"""
# 单次使用：
forces = simple_simulation_step(pose, angle)

# 或者连续使用：
simulator = create_simulator()
try:
    while True:  # 或者其他循环条件
        pose = get_next_pose()  # 获取下一个pose
        angle = get_next_angle()  # 获取下一个angle
        forces = simulator.step(pose, angle)
        # 处理forces...
finally:
    simulator.close()
"""