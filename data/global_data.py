# 创建model用的维度
output_dim=6+1+6+6+512 # 多个视频和两个力的维度
input_dim=6+1# 位姿数据+夹爪数据 维度要和action一样，因为本质上最后是在action的图像上进行模糊和清晰 
# =====cond======
# ResNet18 has output dim of 512
vision_feature_dim = 512
state_feature_dim = 6 # state的维度还有添加的余地，上了动捕再看
global_cond_dim=2*output_dim# 全局条件维度，这里是 位姿数据+夹爪数据+图像特征的维度(512)+左边的力+右边的力
'''action_dim=input_dim
input_dim = action_dim + obs_feature_dim
global_cond_dim = None
if obs_as_global_cond:
    input_dim = action_dim
    global_cond_dim = obs_feature_dim * n_obs_steps
'''
force_feature_dim = 1
# 这两个是数据的维度
# 观察空间的维度
twodim_state_dim = 5
threedim_state_dim = 6+1+force_feature_dim+vision_feature_dim # pose+angle+force+video 
# 动作空间的维度
twodim_action_dim = 2
threedim_action_dim = 6+1+force_feature_dim

# 这两个是预测的维度
# 观察的时间步长
state_horizon = 2
# 动作的时间步长
action_horizon = 8
# 预测的时间步长
pred_horizon = 8
#看2预测8 state为已有的观察的维度，action为动作的维度，相当于是通过观察了预测动作，end to end
data_sum = 25650
train_num = 100
diffusion_num = 100

# agent_pos is 2 dimensional
lowdim_obs_dim = 2
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
twodim_action_dim = 2
zip_data_path ='data/pusht_cchi_v7_replay.zarr.zip'
device ='cuda'