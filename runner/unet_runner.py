import time
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import tqdm
import collections
from data.global_data import *
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from data_analysis.dataset.magiclaw_dataset import create_MagiClaw_dataloader
from diffusion.model.diffusion.cnn_based import create_model
from diffusers.optimization import get_scheduler
from util.plot_visualiazer_util import ActionVisualizer3D
from diffusion.simulation.control_gripper import control_gripper
from diffusion.simulation.gripper_env import GripperEnv
from util.mujoco_util import interpolate_actions
from data_analysis.create_sequence import normalize_data,unnormalize_data
from diffusion.model.repo_from_huggingface import CompatiblePyTorchModelHubMixin
# 2024.11.4 真正的action应该包括了位姿数据加上夹爪的开合数据  那么用来预测的数据应该为?

class UNETRunner(
        nn.Module,
        CompatiblePyTorchModelHubMixin
    ):
    def __init__(self):
        self.model, self.noise_scheduler, self.ema, self.optimizer = create_model()
        self.dataloader,self.dataset=create_MagiClaw_dataloader()
        self.lr_scheduler=get_scheduler(name='cosine',
                                        optimizer=self.optimizer,
                                        num_warmup_steps=500,
                                        num_training_steps=len(self.dataloader))
        self.pretrained=True
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ========= Train  ============
    def compute_loss(self):
        print('-------------Training state_and_vision model-------------')
        with tqdm(range(diffusion_num),desc='Epoch') as tqdm_epoch:
            for epoch in tqdm_epoch:
                epoch_loss=list()
                with tqdm(self.dataloader,desc='Batch', leave=False) as tqdm_batch:#leave=False 进度条不leave,一直在原地刷新
                    for batch in tqdm_batch:#batch为batch_size的个sample_sequence
                        epoch_state=batch['state'].to(self.device).detach().float()
                        epoch_action=batch['action'].to(self.device).detach().float() 
                        '''print(f'epoch_state is {epoch_state.shape}')
                        print(f'epoch_action is {epoch_action.shape}')'''
                        Batch_size = epoch_state.shape[0]
                        # FiLM conditioning
                        # (B, obs_horizon, obs_dim)
                        state_cond = epoch_state[:,:state_horizon,:]#把一个时间步的提取出来，然后flatten
                        # (B, obs_horizon * obs_dim)
                        state_cond = state_cond.flatten(start_dim=1)

                        # 按照action维度生成的噪声
                        noise = torch.randn(epoch_action.shape, device=self.device,dtype=torch.float32)

                        # 将每个数据点采样一个扩散迭代次数 范围为[0, num_train_timesteps-1]，维度为(B,)
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (Batch_size,), device=self.device
                        ).long
                        # 根据每次扩散迭代中的噪声幅度向action添加噪声，使其变模糊（这是前向扩散过程）
                        noisy_actions = self.noise_scheduler.add_noise(
                            epoch_action, noise, timesteps)
                        # 模型预测噪声残差预测值,模型输出的仍然是action的维度
                        noise_pred = self.model(
                            noisy_actions, timesteps, global_cond=state_cond)# output=model(noise,diffusion_iter,obs.flatten(start_dim=1))
                            
                        # 以下添加了损失函数，优化器，还有一个lr_scheduler大抵是优化器吧
                        # 把model预测的噪声和随机生成的噪声进行一个mse计算(毕竟这俩确实维度相同)
                        loss = nn.functional.mse_loss(noise_pred, noise)

                        # optimize
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior(酷)
                        self.lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        self.ema.step(self.model.parameters())

                        # 加载动画条
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tqdm_batch.set_postfix(loss=loss_cpu)
                tqdm_epoch.set_postfix(loss=np.mean(epoch_loss))

                        # 在这之前需要解决所谓步长，空间等的维度问题。还需要解决dataloader中的sample两个函数的问题，
                        # buffer_start_idx, buffer_end_idx,sample_start_idx, sample_end_idx的含义
                        # batch['obs'].shape: torch.Size([256, 2, 5]) 
                        # batch['action'].shape torch.Size([256, 16, 2])
        checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, "cnn_based_model.ckpt")
        self.noise_scheduler.save_pretrained("noise_scheduler_state")
        print('-------------Training state_and_vision model ended-------------')
    # ========= Inference  ============
    def predict_action(self):
        '''
        每次循环预测pred_horizon步的动作序列
        但只执行其中的action_horizon步(通常小于pred_horizon)
        然后重新规划后续动作
        '''
        max_steps=200
        step_idx=0
        print('-----------Testing state_and_vision model-----------')
        # 读取ckpt模型参数
        if self.pretrained==True:
            model_dict = torch.load("cnn_based_model.ckpt", map_location='cuda',weights_only=False)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型参数
        self.model.load_state_dict(model_dict['model_state_dict'])
        self.optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        self.noise_scheduler = DDPMScheduler.from_pretrained("noise_scheduler_state/scheduler_config.json")

        # -------simulation start-------
        state=self.mydataset.initial_state# state就是cond，到时候真实环境情况下就选取显示的initial_state
        state_deque=collections.deque([state]*state_horizon,maxlen=state_horizon)
        done=False
        env=GripperEnv()
        pred_action=np.array([])
        with tqdm(total=200,desc='Output',leave=False) as pbar:
            while not done:
                Batch=1
                state_seq=np.stack(state_deque)
                state_seq=normalize_data(state_seq,self.mydataset.stats['state'])
                state_seq=torch.from_numpy(state_seq).to(self.device, dtype=torch.float32)
                with torch.no_grad():# 禁用梯度计算，节省内存并加快运行速度
                    state_seq=state_seq.unsqueeze(0).flatten(start_dim=1)
                    noise_action=torch.randn((Batch,pred_horizon,input_dim),device=self.device)
                    naction=noise_action
                    self.noise_scheduler.set_timesteps(train_num)
                    for i in range(train_num):
                        noise_pred=self.model(
                            sample=naction,
                            timestep=i,
                            global_cond=state_seq
                        )
                        # noise_scheduler.step() 方法返回的是一个 DDPMSchedulerOutput 对象，而不是直接返回一个张量。所以要加.prev_sample
                        naction=self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=i,
                            sample=naction
                        ).prev_sample 
                naction = naction.detach().to('cpu').numpy()#(1,8,7)
                # action shape is (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=self.mydataset.stats['action'])# 预测的动作出来了，并且将其标准化

                # action_hoizon长度的action
                start = input_dim - 1
                end = start + input_dim # 取出一个现有的state步长的动作
                action = action_pred[start:end,:]
                data=dict()# 最后的action的输出
                data['timestamp']=None
                data['Pose']=None
                data['Angle']=None
                for i in range(len(action)):  # action 的长度相当于是 start:end 的数量
                    current_action = action[i]  # (action_dim,)
                    print(f"执行动作 {step_idx + 1}: {current_action}")
                    pose = current_action[:6]
                    angle = current_action[-1]
                    if data['Angle'] is None:
                        data['Angle']=angle
                        data['Pose']=pose
                    else:
                        data['Angle']=np.concatenate((data['Angle'],angle))# 注意这里的语法：将要连接的数组放在一个列表中 shape:(473, 16)
                        data['Pose']=np.concatenate(data['Pose'],pose)               
                    '''# 进行插值 看着更连续
                    # 获取当前位姿和开度
                    current_pose, _ = env.get_pose()
                    current_angle = env.get_position()
                    # 还可以调整步数
                    interp_steps = 10  
                    interpolated_poses, interpolated_angles = interpolate_actions(current_pose, pose, interp_steps)
                    for interp_pose, interp_angle in zip(interpolated_poses, interpolated_angles):
                        control_gripper(env, interp_pose, interp_angle)
                        step_idx += 1
                        pbar.update(1)

                        if step_idx >= max_steps:
                            done = True
                            break# 如果要插值的话，值会更多，需要遍历插值后的动作序列，然后执行'''
                    
                    step_idx += 1
                    pbar.update(1)
                    if step_idx >= max_steps:
                        done = True
                    if done:
                        break
                    # 更新状态队列（假设执行后更新状态）
                    # state = 获取当前状态的逻辑
                    # state_deque.append(state)
                freq=60
                time_step=1/freq
                timestamp=np.zeros(len(data['pose']))
                for j in range(len(data['pose'])):
                    current_timestamp=time_step*j
                    timestamp[j]=current_timestamp
                data['timestamp']=timestamp
        # 进行仿真
        control_gripper(env, data)
        # 手动关闭渲染器  
        while env.viewer and env.viewer.is_running(): 
            env.render()
            time.sleep(0.01)
        # 仿真完成，关闭渲染器和环境
        env.close()
        print("-------simulation finished-------")

        
        '''for i in range(len(mydataset.episodes_ends)):
            if i==len(mydataset.episodes_ends)-1:
                break
            cur_action=simlation_action[mydataset.episodes_ends[i]:mydataset.episodes_ends[i+1],:]
            pose=cur_action[:6]
            angle=cur_action[-1]
            simple_simulation(pose,angle)'''
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)