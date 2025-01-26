from data_analysis.create_sequece import *
from data_analysis.get_all_data import get_all_data
from tqdm.auto import tqdm

class MagiClawDataset(Dataset):
    def __init__(self, pred_horizon,obs_horizon,action_horizon):
        print(f'-----MagiClawDataset create begin------')
        self.data,self.initial_state=get_all_data()
        self.pred_horizon=pred_horizon
        self.state_horizon=obs_horizon
        self.action_horizon=action_horizon
        # data_dict 的键值有 Depth 、 Video 、 Angle 、 Pose 、 timestamp 、 episodes_ends 、Audio 、L_Force 、R_Force
        self.DepthData=np.array(self.data['Depth'])# data["depth"] shape: (num, 192, 256) 每个值为int
        self.PoseData=np.array(self.data['Pose'])# data["pose"] shape: (num, 6)
        self.episodes_ends=np.array(self.data['episodes_ends'])# data["episodes_ends"] shape: (num, 1)
        self.AngleData=np.array(self.data['Angle'])# data["angle"] shape: (num, 1) 
        self.VideoData=np.array(self.data['Video'])# data["video"] shape: (num,512)
        self.L_ForceData=np.array(self.data['L_Force'])# data["L_Force"] shape: (num, 6)
        self.R_ForceData=np.array(self.data['R_Force'])# data["R_Force"] shape: (num, 6)
        train_data=dict()
        # axis=0 是垂直方向拼接（上下拼接，增加行数）
        # axis=1 是水平方向拼接（左右拼接，增加列数）
        train_data['state']=np.concatenate((self.PoseData, self.AngleData, self.L_ForceData, 
        self.R_ForceData,self.VideoData), axis=1)
        train_data['action']=np.concatenate((self.PoseData, self.AngleData), axis=1)
        stats=dict()
        normalized_data=dict()
        unnormalized_data=dict()
        for key,value in train_data.items():
            stats[key]=get_data_stats(value)
            # print(f'{key} shape is {value.shape}')
            normalized_data[key]=normalize_data(value,stats[key])  
            unnormalized_data[key]=unnormalize_data(value,stats[key])
        indices = create_sample_indices(
            episode_ends=self.episodes_ends,
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
    def __getitem__(self,index):
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
        
def create_MagiClaw_dataloader():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    dataset=MagiClawDataset(pred_horizon=pred_horizon,
                            obs_horizon=state_horizon,
                            action_horizon=action_horizon)
    print(f'------MagiClawDataset create end------')
    print(f'-----dataloader create begin------')
    dataloader=DataLoader(dataset,batch_size=4,num_workers=1,shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True, 
    # don't kill worker process afte each epoch
    persistent_workers=True)
    stats=dataset.stats# 存着每个key的最大值最小值，可以帮助归一化
    print(f'-----dataloader create end------')
    return dataloader,dataset

def test_MagiClaw_dataloader():
    dataloader,dataset=create_MagiClaw_dataloader()
    with tqdm(dataloader,desc='Batch', leave=False) as tqdm_batch:# leave=False 进度条不leave,一直在原地刷新
                for batch in tqdm_batch:# batch为batch_size的个sample_sequence
                    epoch_state=batch['state'].to(device).detach()# 应该的shape为[batch_size,2,512]
                    epoch_action=batch['action'].to(device).detach()# 应该的shape为[batch_size,8,6(pose)+1(angle)]
                    print(f'epoch_state is {epoch_state.shape}')
                    print(f'epoch_action is {epoch_action.shape}')