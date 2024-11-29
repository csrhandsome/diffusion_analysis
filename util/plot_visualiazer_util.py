import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.interpolate import make_interp_spline

class ActionVisualizer3D:
    def __init__(self, action_limits=(-0.75, 0.75)):
        plt.style.use('default')
        
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴范围
        self.ax.set_xlim(action_limits)
        self.ax.set_ylim(action_limits)
        self.ax.set_zlim(action_limits)
        
        # 设置图表样式
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title('Robot Position 3D Trajectory', fontsize=14, pad=15)
        self.ax.set_xlabel('X Position', fontsize=12)
        self.ax.set_ylabel('Y Position', fontsize=12)
        self.ax.set_zlabel('Z Position', fontsize=12)
        
        # 初始化轨迹线和当前点
        self.trajectory, = self.ax.plot([], [], [], 'b-', alpha=0.6, label='Path', linewidth=2)
        self.current_point, = self.ax.plot([], [], [], 'ro', markersize=12, label='Current Position')
        
        # 存储历史轨迹
        self.history_x = []
        self.history_y = []
        self.history_z = []
        
        # 存储插值后的轨迹点
        self.smooth_x = []
        self.smooth_y = []
        self.smooth_z = []
        
        # 存储当前文本标注
        self.current_text = None
        
        # 添加图例
        self.ax.legend(loc='upper right')
        
        # 设置背景样式
        self.fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        # 创建动画对象
        self.anim = None
        self.current_action = None
        
    def smooth_trajectory(self):
        if len(self.history_x) > 3:
            # 创建平滑插值点
            points = np.array([self.history_x, self.history_y, self.history_z]).T
            # 生成插值参数
            t = np.linspace(0, 1, len(points))
            t_smooth = np.linspace(0, 1, len(points) * 5)
            
            try:
                # 对每个维度分别进行样条插值
                k = min(3, len(points)-1)
                spl_x = make_interp_spline(t, points[:, 0], k=k)
                spl_y = make_interp_spline(t, points[:, 1], k=k)
                spl_z = make_interp_spline(t, points[:, 2], k=k)
                
                self.smooth_x = spl_x(t_smooth)
                self.smooth_y = spl_y(t_smooth)
                self.smooth_z = spl_z(t_smooth)
            except:
                # 如果插值失败，使用原始点
                self.smooth_x = self.history_x
                self.smooth_y = self.history_y
                self.smooth_z = self.history_z
        else:
            self.smooth_x = self.history_x
            self.smooth_y = self.history_y
            self.smooth_z = self.history_z

    def animate(self, frame):
        # 更新平滑轨迹
        self.smooth_trajectory()
        
        # 更新轨迹线
        self.trajectory.set_data(self.smooth_x, self.smooth_y)
        self.trajectory.set_3d_properties(self.smooth_z)
        
        # 更新当前点
        if self.current_action is not None:
            self.current_point.set_data([self.current_action[0]], [self.current_action[1]])
            self.current_point.set_3d_properties([self.current_action[2]])
            
            # 更新文本标注
            if self.current_text is not None:
                self.current_text.remove()
            self.current_text = self.ax.text(
                self.current_action[0], 
                self.current_action[1], 
                self.current_action[2] + 0.05,
                f'({self.current_action[0]:.2f}, {self.current_action[1]:.2f}, {self.current_action[2]:.2f})',
                ha='center', 
                va='bottom'
            )
        
        # 动态调整视角
        self.ax.view_init(elev=20, azim=frame % 360)
        
        return self.trajectory, self.current_point

    def update(self, current_action):
        self.current_action = current_action
        
        # 更新历史轨迹
        self.history_x.append(current_action[0])
        self.history_y.append(current_action[1])
        self.history_z.append(current_action[2])
        
        # 如果动画还没有开始，就创建动画
        if self.anim is None:
            self.anim = FuncAnimation(
                self.fig,
                self.animate,
                interval=20,  # 20ms的更新间隔
                blit=True,
                cache_frame_data=False
            )
        
        plt.draw()
        plt.pause(0.001)
    
    def close(self):
        if self.anim is not None:
            self.anim.event_source.stop()
        plt.close(self.fig)