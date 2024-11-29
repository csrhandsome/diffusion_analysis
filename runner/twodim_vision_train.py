from data_analysis.dataset.demo_push_t_image_dataset import *
from diffusion.model.cnn_based import *
from diffusion.model.visionencoder import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def twodim_vision_create_model():
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet('resnet18')

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder).to(device)
    
    # create network object 记住两个都要to(device)，to(device)可以把数据类型变成torch.cuda.FloatTensor
    model = ConditionalUnet1D(
        input_dim=twodim_action_dim,
        global_cond_dim=obs_dim*state_horizon
    ).to(device)
    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': model
    })
    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )
    ema = EMAModel(parameters=nets.parameters(),power=0.75)
    optimizer = torch.optim.AdamW(params=nets.parameters(),lr=1e-4, weight_decay=1e-6)
    return nets,noise_scheduler,ema,optimizer

def twodim_vision_test_dataloader():
    dataloader,dataset=create_demo_pusht_image_dataloader()
    # 测试数据加载
    try:
        for i, batch in enumerate(dataloader):
            print(f"Successfully loaded batch {i}")
            print(f"Batch keys: {batch.keys()}")
            print(f"Batch shapes: {[(k, v.shape) for k, v in batch.items()]}")
            break
    except Exception as e:
        print("Error loading batch:", e)
        import traceback
        traceback.print_exc()
    
    return dataloader, dataset

def twodim_vision_test_dataset():
    dataloader,dataset=create_demo_pusht_image_dataloader()
    print(f"Dataset length: {len(dataset)}")  # 检查数据集大小
    
    # 测试能否从数据集直接取出一个样本
    try:
        sample = dataset[0]
        print("Sample keys:", sample.keys())
    except Exception as e:
        print("Error getting sample:", e)

#**Training**
#Takes about 2.5 hours. 
#@markdown to load pre-trained weights
def twodim_vision_train_model():
    print("Training vision model...")
    nets,noise_scheduler,ema,optimizer=twodim_vision_create_model()
    dataloader,dataset=create_demo_pusht_image_dataloader()
    num_epochs = 100
    lr_scheduler = get_scheduler(name='cosine',optimizer=optimizer,num_warmup_steps=500,num_training_steps=len(dataloader)*num_epochs)
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:#传入可以迭代的对象 dataloader,range()等会自动更新，否则要自己update
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    # x: [batch_size, num_frames, channels, height, width]
                    # image shape is torch.Size([64, 2, 3, 96, 96])
                    nimage = nbatch['image'][:,:state_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:,:state_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B,T= nagent_pos.shape[:2] # batch_size, num_frames

                    # encoder vision features
                    # end_dim=1表示将从维度0到维度1进行展平
                    # 也就是将batch_size和num_frames这两个维度合并
                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        nimage.shape[0],nimage.shape[1],-1)
                    # (B,obs_horizon,D)

                    # concatenate 视频feature and 低维state
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # 向前传播
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = nets['noise_pred_net'](
                        noisy_actions, timesteps, global_cond=obs_cond)

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
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()#返回一个int or float
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)#这个相当于是在进度条上面显示loss(额外信息)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model
    # is used for inference
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())
    checkpoint = {
    'model_state_dict': nets['noise_pred_net'].state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, "cnn_based_vision_model.ckpt")
    noise_scheduler.save_pretrained("noise_scheduler_vision")
    return nets,noise_scheduler,ema,optimizer


# **Test**
# limit enviornment interaction to 200 steps before termination
def twodim_vision_test_model():
    # 读取数据
    dataloader,mydataset=create_demo_pusht_image_dataloader()
    
    # 读取ckpt模型参数
    model_dict = torch.load("cnn_based_vision_model.ckpt", map_location='cuda',weights_only=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建模型
    nets,noise_scheduler,ema,optimizer=twodim_vision_create_model()
    # 加载模型参数
    nets['noise_pred_net'].load_state_dict(model_dict['model_state_dict'])
    optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    noise_scheduler = DDPMScheduler.from_pretrained("noise_scheduler_vision")
    max_steps = 200
    model=nets['noise_pred_net']
    state=np.array((100,200,150,250,1.5707963267948966))#这个为初始的state 
    state_deque=collections.deque([state]*state_horizon,maxlen=state_horizon)


    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:# 不能自己迭代，需要手动更新
        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in state_deque])
            agent_poses = np.stack([x['agent_pos'] for x in state_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=mydataset.stats['agent_pos'])
            # images are already normalized to [0,1]
            nimages = images

            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                image_features = model['vision_encoder'](nimages)
                # (2,512)
                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, twodim_action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(diffusion_num)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = model['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=mydataset.stats['action'])

            # only take action_horizon number of actions
            start = state_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                state, reward, done, _, info = env.step(action[i])
                # save observations
                state_deque.append(state)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    '''# visualize
    from IPython.display import Video
    vwrite('vision_push_demo.mp4', imgs)
    Video('vision_push_demo.mp4', embed=True, width=256, height=256)'''

