import os
import pandas as pd
import numpy as np
from util.pose_transform_util import *
# 可以根据深度数据中的timestamp来截取视频中的frame吗，这样就可以实现一对一的对应
def load_depth_data(first_dir= 'data/drawCircle')->dict: 
    """
    加载深度数据
    
    Args:
        first_dir (str): 数据目录路径
        
    Returns:
        numpy.ndarray: 深度数据数组，形状为 (batch, height, width)
    """
    Depth = None
    
    # 检查输入路径是文件还是目录
    if os.path.isfile(first_dir):
        print(f"警告: {first_dir} 是文件而不是目录，无法加载深度数据")
        return np.array([])  # 返回空数组
    
    # 递归查找所有 .bin 文件
    for root, dirs, files in os.walk(first_dir):
        if "__MACOSX" in dirs:
            dirs.remove("__MACOSX")  # 跳过 __MACOSX 目录
            
        for file in files:
            if file.endswith('.bin') and 'Depth' in root:
                filepath = os.path.join(root, file)
                try:
                    # 注意 numpy 是行优先，所以是 (height, width) 实际上是 256*192
                    DepthData = np.fromfile(filepath, dtype=np.uint16).reshape(192, 256) / 10000
                    
                    if Depth is None:
                        Depth = DepthData
                    else:
                        Depth = np.concatenate((Depth, DepthData))
                except Exception as e:
                    print(f"加载文件 {filepath} 时出错: {e}")
    
    # 如果没有找到任何深度数据
    if Depth is None:
        print(f"警告: 在 {first_dir} 中没有找到任何深度数据")
        return np.array([])  # 返回空数组
    
    Depth = Depth.reshape(-1, 192, 256)  # (batch, height, width)
    return Depth



