from data_analysis.dataset.magiclaw_dataset import *
from diffusion.model.cnn_based import *
from util.plot_visualiazer_util import *
from diffusion.simulation.replay_simulation import *
# 2024.11.4 真正的action应该包括了位姿数据加上夹爪的开合数据  那么用来预测的数据应该为?
def threedim_state_and_vision_create_model():
    model = ConditionalUnet1D(input_dim=input_dim, global_cond_dim=global_cond_dim)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''noise = torch.randn((1, 8, 2))#观测两个现有的时间维度,预测8个时间维度
    state=torch.zeros((1,2,5))
    diffusion_iter=torch.zeros((1,))
    output=model(noise,diffusion_iter,state.flatten(start_dim=1))
    #print(f'output is noise is {output}')
    denoised=noise-output'''# 演示
    num_diffuison=100
    noise_scheduler=DDPMScheduler(num_train_timesteps=num_diffuison,
                                  clip_sample=True,
                                  prediction_type='epsilon',
                                  beta_schedule='squaredcos_cap_v2')
    model=model.to(device)
    ema=EMAModel(parameters=model.parameters(),power=0.75)
    optimizer=torch.optim.AdamW(params=model.parameters(),lr=1e-4,weight_decay=1e-6)
    print(f'model created')
    return model,noise_scheduler,ema,optimizer

def threedim_state_and_vision_train_model():
    print('-------------Training state model-------------')
    model,noise_scheduler,ema,optimizer=threedim_state_and_vision_create_model()
    dataloader,dataset=create_MagiClaw_dataloader()
    lr_scheduler=get_scheduler(name='cosine',optimizer=optimizer,num_warmup_steps=500,num_training_steps=len(dataloader))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with tqdm(range(diffusion_num),desc='Epoch') as tqdm_epoch:
        for epoch in tqdm_epoch:
            epoch_loss=list()
            with tqdm(dataloader,desc='Batch', leave=False) as tqdm_batch:#leave=False 进度条不leave,一直在原地刷新
                for batch in tqdm_batch:#batch为batch_size的个sample_sequence
                    epoch_state=batch['state'].to(device).detach().float()#epoch_state is torch.Size([4, 2, 6])
                    epoch_action=batch['action'].to(device).detach().float()#epoch_action is torch.Size([4, 8, 7])  
                    '''print(f'epoch_state is {epoch_state.shape}')
                    print(f'epoch_action is {epoch_action.shape}')'''
                    Batch_size = epoch_state.shape[0]
                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    state_cond = epoch_state[:,:state_horizon,:]#把一个时间步的提取出来，然后flatten
                    # (B, obs_horizon * obs_dim)
                    state_cond = state_cond.flatten(start_dim=1)

                    # 按照action形状生成噪声,只用到了形状
                    noise = torch.randn(epoch_action.shape, device=device,dtype=torch.float32)

                    # 将每个数据点采样一个扩散迭代次数 范围为[0, num_train_timesteps-1]，维度为(B,)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (Batch_size,), device=device
                    ).long()
                    # 根据每次扩散迭代中的噪声幅度向原始图像添加噪声
                    # （这是前向扩散过程）
                    noisy_actions = noise_scheduler.add_noise(
                        epoch_action, noise, timesteps)
                    # 噪声残差预测值
                    noise_pred = model(
                        noisy_actions, timesteps, global_cond=state_cond)# output=model(noise,diffusion_iter,obs.flatten(start_dim=1))
                    # 以下添加了损失函数，优化器，还有一个lr_scheduler大抵是优化器吧
                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(model.parameters())

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
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, "cnn_based_model.ckpt")
    noise_scheduler.save_pretrained("noise_scheduler_state")
    return model,noise_scheduler,ema,optimizer

def threedim_state_and_vision_test_model():
    # 读取数据
    dataloader,mydataset=create_MagiClaw_dataloader()
    max_steps=200
    step_idx=0
    # 读取ckpt模型参数
    model_dict = torch.load("cnn_based_model.ckpt", map_location='cuda',weights_only=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建模型
    model,noise_scheduler,ema,optimizer=threedim_state_and_vision_create_model()
    
    # 加载模型参数
    model.load_state_dict(model_dict['model_state_dict'])
    optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    noise_scheduler = DDPMScheduler.from_pretrained("noise_scheduler_state")
    state=mydataset.initial_state
    state_deque=collections.deque([state]*state_horizon,maxlen=state_horizon)
    done=False
    with tqdm(total=200,desc='Output',leave=False) as pbar:
        while not done:
            Batch=1
            state_seq=np.stack(state_deque)
            state_seq=normalize_data(state_seq,mydataset.stats['state'])
            state_seq=torch.from_numpy(state_seq).to(device, dtype=torch.float32)
            with torch.no_grad():# 禁用梯度计算，节省内存并加快运行速度
                state_seq=state_seq.unsqueeze(0).flatten(start_dim=1)
                noise_action=torch.randn((Batch,pred_horizon,input_dim),device=device)
                naction=noise_action
                noise_scheduler.set_timesteps(train_num)
                for i in range(train_num):
                    noise_pred=model(
                        sample=naction,
                        timestep=i,
                        global_cond=state_seq
                    )
                    # noise_scheduler.step() 方法返回的是一个 DDPMSchedulerOutput 对象，而不是直接返回一个张量。所以要加.prev_sample
                    naction=noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=i,
                        sample=naction
                    ).prev_sample 
            naction = naction.detach().to('cpu').numpy()#(1,8,7)
            # action shape is (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=mydataset.stats['action'])#预测的动作出来了，并且将其标准化

            # action_hoizon长度的action
            start = input_dim - 1
            end = start + input_dim #取出一个现有的state步长的动作
            action = action_pred[start:end,:]
            '''for i in range(len(action)):
                print(f'{i}th action is {action[i]}')'''
            for i in range(len(action)): # action的长度相当于是start:end的数量
                current_action = action[i]#(2,)
                pose=current_action[:6]
                angle=current_action[-1]
                simple_simulation(pose,angle)