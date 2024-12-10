from data_analysis.get_all_data import *
from diffusion.model.visionencoder import test_vision_encoder
from data_analysis.dataset.magiclaw_dataset import create_MagiClaw_dataloader,test_MagiClaw_dataloader
from diffusion.simulation.control_gripper import control_gripper
if __name__=='__main__':
   # data,initial_state=get_all_data()
   # 定义一系列带有时间戳的动作
   control_gripper(actions=[
      {"timestamp": 0.0, "pose": np.array([0.1, 0.0, 0.0, 0.0, 0.0, np.pi/4]), "angle": 0.05},
      {"timestamp": 0.33, "pose": np.array([0.2, 0.0, 0.0, 0.0, 0.0, np.pi/2]), "angle": 0.03},
      {"timestamp": 0.66, "pose": np.array([0.3, 0.0, 0.0, 0.0, 0.0, 3*np.pi/4]), "angle": 0.07},
   ])
   #test_MagiClaw_dataloader()
   #print(cv2.__version__)
   #test_videodict()