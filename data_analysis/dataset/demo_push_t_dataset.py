from data_analysis.create_sequece import *



class OthersDataset(Dataset):
    def __init__(self, data_path,pred_horizon,obs_horizon,action_horizon):
        self.data=zarr.open(data_path,mode='r')
        self.pred_horizon=pred_horizon
        self.state_horizon=obs_horizon
        self.action_horizon=action_horizon
        # batch['obs'].shape: torch.Size([batch_size, 2, 5])
        # batch['action'].shape torch.Size([batch_size, 16, 2])
        self.action=np.array(self.data['data']['action'])
        self.state=np.array(self.data['data']['state'])
        self.episode_ends=np.array(self.data['meta']['episode_ends'])
        train_data=dict()
        train_data['action']=self.action
        train_data['state']=self.state
        stats=dict()
        normalized_data=dict()
        unnormalized_data=dict()
        for key,value in train_data.items():
            stats[key]=get_data_stats(value)
            normalized_data[key]=normalize_data(value,stats[key])  
            unnormalized_data[key]=unnormalize_data(value,stats[key])
        indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=self.pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=self.state_horizon-1,
            pad_after=self.action_horizon-1)
        self.indices=indices
        self.train_data=train_data#len(train_data)=2 which is not the same as len(indices)
        self.normalized_data=normalized_data#数据一切正常
        self.unnormalize_data=unnormalize_data
        self.stats=stats
    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):#目前问题：无法生成一个batch ,    nsample为空
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[index]#indices一个就是四个值
        # sampel_sequence保留了原有的key
        nsample = sample_sequence(
            train_data=self.normalized_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        # discard unused observations
        nsample['state'] = nsample['state'][:self.state_horizon,:]
        # print(f'nsamppe["state"] is {nsample["state"]}')
        # print(f'nsamppe["action"] is {nsample["action"]}')
        return nsample



def create_demo_pusht_dataloader():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    dataset=OthersDataset(data_path=zip_data_path,pred_horizon=pred_horizon,obs_horizon=state_horizon,action_horizon=action_horizon)
    dataloader=DataLoader(dataset,batch_size=256,num_workers=1,shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True, 
    # don't kill worker process afte each epoch
    persistent_workers=True)
    stats=dataset.stats#存着每个key的最大值最小值，可以帮助归一化
    print(f'dataloader created')
    return dataloader,dataset