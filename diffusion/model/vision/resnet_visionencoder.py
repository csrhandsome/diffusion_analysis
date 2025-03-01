#@markdown ### **Vision Encoder**
#@markdown
#@markdown Defines helper functions:
#@markdown - `get_resnet` to initialize standard ResNet vision encoder
#@markdown - `replace_bn_with_gn` to replace all BatchNorm layers with GroupNorm
import torch
import torch.nn as nn
import torchvision
from diffusion.model.diffusion.cnn_based import *
from data_analysis.dataset.demo_push_t_image_dataset import PushTImageDataset
from data.global_data import *
from typing import Callable
#养成好习惯，每个形参写类型，返回值写类型，必要时候用callable来定义函数

# **kwargs这是一个特殊的参数，允许函数接受任意数量的关键字参数这些额外的参数会被收集到一个字典中
def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # for resnet18, the output dim should be 512
    #通过这种修改，ResNet模型的最后一层被替换成nn.Identity(),将直接输出其最后一个卷积层的特征，而不是进行分类预测。
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],#Callable[[Arg1Type, Arg2Type, ...], ReturnType]
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(# 在Python中，对象（如 nn.Module）是通过引用传递的
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),# 等于形参是x，return一个isinstance(x, nn.BatchNorm2d)
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

def test_vision_encoder():
    # 初始化编码器
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    
    # 创建测试输入
    batch_size = 32
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # 前向传播
    with torch.no_grad():
        output = vision_encoder(input_tensor)
    
    # 打印shape信息
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # 验证输出维度是否正确
    assert output.shape == (batch_size, 512)
    
    return True

