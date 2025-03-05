from data_analysis.dataset.vla_singlehand_dataset import MagiclawVLADataset
from train.consumer_dataset import VLAConsumerDataset
from train.rdt_train import train
if __name__=='__main__':
   '''ds = MagiclawVLADataset()
   for i in range(len(ds)):
      print(f"Processing episode {i}/{len(ds)}...")
      ds.get_item(i)'''
   train()