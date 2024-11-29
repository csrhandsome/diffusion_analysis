from data_analysis.get_all_data import *
from diffusion.model.visionencoder import test_vision_encoder
from data_analysis.dataset.magiclaw_dataset import create_MagiClaw_dataloader,test_MagiClaw_dataloader
from diffusion.simulation.replay_simulation import simple_simulation
if __name__=='__main__':
   data,initial_state=get_all_data()
   #simple_simulation(data)
   #test_MagiClaw_dataloader()
   #print(cv2.__version__)
   #test_videodict()