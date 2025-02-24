import re
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import tqdm
import collections
from data_analysis.dataset.magiclaw_dataset import create_MagiClaw_dataloader
from diffusion.model.diffusion.cnn_based import ConditionalResidualBlock1D, ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from data_analysis.create_sequence import normalize_data,unnormalize_data
from diffusion.model.repo_from_huggingface import CompatiblePyTorchModelHubMixin
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

class UNETRunner(
        nn.Module,
        CompatiblePyTorchModelHubMixin
    ):
    def __init__(self, action_dim, pred_horizon,config, n_obs_steps,
                 img_dim, state_dim, 
                 img_cond_len,obs_as_global_cond=True,dtype=torch.bfloat16):
        # init model
        self.pred_horizon,self.img_cond_len = pred_horizon,img_cond_len
        self.pretrained = True
        self.obs_as_global_cond = obs_as_global_cond
        noise_scheduler_config = config['noise_scheduler']
        input_dim = action_dim + state_dim + img_dim
        global_cond_dim = None
        # 有 n_obs_steps 个历史观测步，模型需要将所有时间步的特征 拼接成一个长向量
        if obs_as_global_cond:# 暂定为True,因为要输入输出是一样的
            input_dim = action_dim
            global_cond_dim = (state_dim + img_dim)*n_obs_steps
        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=256,
            down_dims=[256,512,1024],
            kernel_size=5,
            n_groups=8
        )
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        noise_scheduler=DDPMScheduler(num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
                                    clip_sample=True,
                                    prediction_type=noise_scheduler_config['prediction_type'],
                                    beta_schedule=noise_scheduler_config['beta_schedule'])
        self.model=self.model.to(self.device)
        self.ema=EMAModel(parameters=self.model.parameters(),power=0.75)
        self.optimizer=torch.optim.AdamW(params=self.model.parameters(),lr=1e-4,weight_decay=1e-6)
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
                num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
                beta_schedule=noise_scheduler_config['beta_schedule'],
                prediction_type=noise_scheduler_config['prediction_type'],
            )
        self.lr_scheduler = get_scheduler(name='cosine',
                                        optimizer=self.optimizer,
                                        num_warmup_steps=500,
                                        num_training_steps=len(self.dataloader))
        self.prediction_type = noise_scheduler_config['prediction_type']
        print(f'model created')
        
        # init data
        self.hidden_size = config['rdt']['hidden_size']
        self.dataloader,self.dataset = create_MagiClaw_dataloader()
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'], 
            in_features = img_dim, 
            out_features = self.hidden_size)
        action_mask_dim=action_dim
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'], 
            in_features = state_dim,# unet里面不合并了，因为没有注意力机制，并且将video和state作为了globalcond,但是所有的输入之前都要concat一下action_mask,action_mask可以用来指导生成。
            out_features = self.hidden_size
        )
        self.all_cond_adaptor = self.build_condition_adapter(
            config['all_cond_adaptor'], # 也许需要更大维度的MLP
            in_features = state_dim+action_mask_dim+img_dim,
            out_features = self.hidden_size
        )
        print(f'cond_adapter created')# 映射到统一的隐藏空间(hidden_size)的模型建立了
    
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        '''
        将不同模态的输入映射到统一的隐藏空间的模型(MLP还有Linear)
        '''
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    def adapt_conditions(self, img_tokens, state_tokens,action_mask):
        '''
        将语言、图像和状态的条件输入通过适配器映射到统一的隐藏空间
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, state_len, state_token_dim)
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)
        adpated_action_mask = self.state_adaptor(action_mask)# 试着共用一下state_adaptor,毕竟可能在同一个空间
        return adpated_img, adpated_state, adpated_action_mask
    
    def conditional_sample(self,  img_cond, 
                        state_traj, action_mask, ctrl_freqs):
        '''
        生成action
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
            dtype=dtype, device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
    
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = noisy_action
            action_traj = self.state_adaptor(action_traj)
            img_cond, state_cond, action_mask_cond = self.adapt_conditions(
                img_cond, state_traj, action_mask)# (B,img_len,hidden_size),(B,1,hidden_size),(B,1,hidden_size)
            # Predict the model output t不行就不unsqueeze
            global_cond = torch.cat([img_cond, state_cond, action_mask_cond], dim=1)# 三个cond
            model_output = self.model(noisy_action, t, global_cond=global_cond)
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action
    
    # ========= Train  ============
    def compute_loss(self,img_tokens, state_tokens,
                     action_gt,action_mask,ctrl_freqs,lang_tokens=None):
        '''
        把action变成噪声,然后预测噪声
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        batch_size = img_tokens.shape[0]
        # device = img_tokens.device  

        # Sample noise that we'll add to the actions
        noise = torch.randn(
            action_gt.shape, dtype=action_gt.dtype, device=self.device
        )
        # 将每个数据点采样一个扩散迭代次数 范围为[0, num_train_timesteps-1]，维度为(B,)
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        # 根据每次扩散迭代中的噪声幅度向action添加噪声，使其变模糊（这是前向扩散过程）
        noisy_action = self.noise_scheduler.add_noise(
            action_gt, noise, timesteps)
        
        # ======global_cond的处理======
        if self.obs_as_global_cond:
            img_cond, state_cond, action_mask_cond = self.adapt_conditions(
                img_tokens, state_tokens,action_mask)# (B,img_len,hidden_size),(B,1,hidden_size),(B,1,hidden_size)
            
            global_cond = torch.cat([img_cond, state_cond, action_mask_cond], dim=1)# 三个cond
            
            pred = self.model(noisy_action, timesteps, global_cond=global_cond)
        else:# 局部条件情况 将cond_data和action合并
            # 合并state和action，形成一个大的state_action_traj
            state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
            # Append the action mask to the input sequence
            action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
            cond_data = torch.cat([state_action_traj, action_mask, img_cond], dim=2)

            # 映射到统一的hidden_size
            cond_data = self.all_cond_adaptor(cond_data)

            pred = self.model(cond_data, timesteps, global_cond=None)
        pred_type = self.prediction_type 
        if pred_type == 'epsilon':# epsilon是预测的噪声
            target = noise
        elif pred_type == 'sample':# sample是预测的action
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target)
        return loss
    
    # ========= Inference  ============
    def predict_action(self,img_tokens, state_tokens,
                       action_mask,ctrl_freqs,lang_tokens=None):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_mask: (batch_size, 1, action_dim),
            which should be a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        img_cond, state_traj = self.adapt_conditions(
            img_tokens, state_tokens)
        
        # Run sampling
        action_pred = self.conditional_sample(
            img_cond, 
            state_traj, action_mask, ctrl_freqs,
        )
        
        return action_pred
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)