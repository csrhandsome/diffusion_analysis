from util.pose_transform_util import *

def load_episodes_data(data):
    """
    根据时间戳数组找出每个episode的结束索引
    适用于每个episode重新从0开始计时的情况
    参数:
    timestamps: numpy数组，每个episode从0开始的时间戳
    返回:
    episode_ends: numpy数组，存储每个episode结束时的索引
    """
    timestamps=data["timestamp"] 
    # 保存为CSV
    np.savetxt('timestamps.csv', timestamps, delimiter=',', fmt='%f')
    # 计算时间差
    time_diffs = np.diff(timestamps)
    
    # 找出时间戳突然变小的位置（说明是新的episode开始）
    episode_breaks = np.where(time_diffs < 0)[0]
    
    # episode_breaks找到的是间隔位置，需要+1才是episode的结束索引
    episode_ends = episode_breaks + 1
    
    # 确保包含最后一个时间点
    if len(episode_ends) == 0 or episode_ends[-1] != len(timestamps):
        episode_ends = np.append(episode_ends, len(timestamps))
    data['episodes_ends']=episode_ends
    # 保存为CSV
    # np.savetxt('episode_ends.csv', episode_ends, delimiter=',', fmt='%f')
    return data