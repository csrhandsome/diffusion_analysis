import numpy as np


# PID (比例-积分-微分) 控制器是一种常用的反馈控制系统，用于使系统输出达到期望值。
class PIDController:
    def __init__(self, kp=1.0, ki=0.1, kd=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = np.zeros(6)
        self.integral = np.zeros(6)
        
        # 为位置和姿态设置不同的增益
        self.gain_matrix = np.array([
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]  # 位置和姿态的增益比例
        ]).T

    def compute(self, current_pose, target_pose, dt):
        error = target_pose - current_pose
        
        # 处理角度误差，确保在-pi到pi之间
        error[3:] = (error[3:] + np.pi) % (2 * np.pi) - np.pi

        # 积分项
        self.integral += error * dt
        # 防止积分饱和
        max_integral = 1.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)

        # 微分项
        derivative = (error - self.prev_error) / dt

        # 计算PID输出
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        # 应用增益矩阵
        output = output * self.gain_matrix.flatten()

        self.prev_error = error.copy()
        return output