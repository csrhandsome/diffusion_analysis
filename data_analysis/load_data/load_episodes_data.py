import os
import numpy as np
import pandas as pd

def load_episodes_length(first_dir):
    '''
    原始pose涉及到位姿的转换,根据state的shape来计算episode_lens
    '''
    Pose=None
    file_paths=[]
    episode_lens = []
    for root, dirs, files in os.walk(first_dir):
        if "__MACOSX" in dirs:
            dirs.remove("__MACOSX")  # 跳过__MACOSX目录
        for file in files:
            if file == 'PoseData.csv':  # 直接检查文件名
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                total_data=np.array(pd.read_csv(file_path,skiprows=1))# 数据的第一行忽略
                cur_pose=np.array(total_data[:,1:])
                if Pose is None:
                    Pose=cur_pose
                else:
                    Pose=np.concatenate((Pose,cur_pose))# shape:(473, 16)
                _len = cur_pose.shape[0]
                episode_lens.append(_len)           
    return episode_lens

