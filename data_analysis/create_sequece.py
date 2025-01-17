from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import zarr
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from util.pose_transform_util import *
from data.global_data import *


def get_data_stats(data)->dict():
    '''如果原始数据形状是(10, 20, 30, 5),
    那么reshape后的形状将变为(6000, 5)'''
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    '''利用上面返回的字典 stats 手动进行归一化'''
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    '''别人的sequence函数,用于dataset类的调用getitem的时候使用
计算 buffer_start_idx 和 buffer_end_idx,这是从原始数据中实际提取的范围
计算 sample_start_idx 和 sample_end_idx,这是提取的数据在最终样本中的位置。'''
    result = dict()
    for key, input_arr in train_data.items():#保留了原有的key，只对数据做了处理
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    #print("sample_sequence created and used")
    return result


def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    '''别人的创建indices函数   episode_ends是一个int数组,表示每个episode的结束(时间还是位移?)
# 问题在于返回的四个形参太大了，超出了索引的范围很多了
# indices is [[     0      7      1      8]
#  [     0      8      0      8]
#  [     1      9      0      8]
#  ...
#  [358569 358572      0      3]
#  [358570 358572      0      2]
#  [358571 358572      0      1]]'''
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx#每个段的长度
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        # range stops one idx before end
        # 计算 buffer_start_idx 和 buffer_end_idx，这是从原始数据中实际提取的范围
        # 计算 sample_start_idx 和 sample_end_idx，这是提取的数据在最终样本中的位置。
        for idx in range(int(min_start), int(max_start)+1): 
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)#就是idx
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx#就是idx+sequence_length-episode_length
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    #print(f'indices is: {indices}')
    return indices


def my_create_sample_indices(episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    '''根据时间步，reshape数据'''
    return

