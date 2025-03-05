import os
import pandas as pd
import numpy as np
import fnmatch
from util.pose_transform_util import mat_to_pose

def load_gripper_pose_data(first_dir= 'data/drawCircle')->dict: 
    '''
    原始pose涉及到位姿的转换
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
    matrix=Pose
    pose_matrix=np.array([])
    for i in range(matrix.shape[0]):
        #print(f"\n处理第{i+1}个矩阵:")
        single_matrix = matrix[i]
        #print(f'矩阵{i}的原始形状: {single_matrix.shape}')
        # 重塑为4x4矩阵
        single_matrix = single_matrix.reshape(4, 4)
        #不要忘记转置
        single_matrix=np.transpose(single_matrix)
        #print(f'重塑后的矩阵:{single_matrix}')
        # 转换为6维姿态向量
        cur_pose = mat_to_pose(single_matrix)
        #print("6维姿态向量:", pose)
        if pose_matrix.size == 0:
            pose_matrix = cur_pose
        else:
            pose_matrix = np.vstack((pose_matrix,cur_pose))#vstack会自动处理维度,concatenate需要手动确保维度匹配             
    
    Pose=pose_matrix
    return Pose,episode_lens



