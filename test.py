from data_analysis.dataset.vla_singlehand_dataset import VLADataset
from train.consumer_dataset import VLAConsumerDataset
from train.rdt_train import train
from useless.test_cuda import test_cuda
from runner.unet_runner import UNETRunner
import yaml
import torch
from data_analysis.get_all_data import convert_all_to_hdf5
if __name__=='__main__':
   ds = VLADataset()
   for i in range(len(ds)):
      print(f"Processing episode {i}/{len(ds)}...")
      sample = ds.get_item()
      print(sample)
   # convert_all_to_hdf5('data/drawCircle')
   # test_cuda()
#    with open('configs/config.yaml', "r") as fp:
#         config = yaml.safe_load(fp)
#    img_cond_len = (config["common"]["img_history_size"] 
#                             * config["common"]["num_cameras"] 
#                             * 1)
#    unet = UNETRunner(
#             action_dim=config["common"]["state_dim"],
#             lan_dim=config["model"]["lang_token_dim"],
#             pred_horizon=config["common"]["action_chunk_size"],# 64
#             config=config["model"],
#             img_dim=config["model"]["img_token_dim"],# 1152
#             state_dim=config["model"]["state_token_dim"],
#             max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
#             img_cond_len=img_cond_len,
#             obs_as_global_cond=True,
#             dtype=torch.bfloat16,
#         )

