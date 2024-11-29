import os
import pandas as pd
import numpy as np
from util.pose_transform_util import *
#在这里我要更新一个真正的episodes_ends
def load_dataframe_matrix(first_dir= 'data/drawCircle'):
    data=dict()
    Pose_columns=['Timestamp','NC','NB','NA','N','OC','OB','OA','O','AC','AB','AA','A','PC','PB','PA','P']  
    data['PoseData']=pd.DataFrame(columns=Pose_columns)
    first_filenames=os.listdir(first_dir)
    for filename1 in first_filenames:
        if filename1.endswith('__MACOSX'):
            continue
        else:
            second_dir=os.path.join(first_dir,filename1)
            second_filenames=os.listdir(second_dir)
            for filename2 in second_filenames:
                filepath=os.path.join(second_dir,filename2)
                if filename2.endswith('PoseData.csv'):
                    PoseData=pd.read_csv(filepath,skiprows=1)#数据的第一行忽略，自己写coloums
                    PoseData.columns=Pose_columns
                    data['PoseData']=pd.concat([data['PoseData'],PoseData],ignore_index=True)
                '''elif filename2.endswith('AngleData.csv'):
                    AngleData=pd.read_csv(filepath)
                    data['AngleData']=pd.concat([data['AngleData'],AngleData],ignore_index=True)
                elif filename2.endswith('L_ForceData.csv'):
                    L_ForceData=pd.read_csv(filepath)
                    data['L_ForceData']=pd.concat([data['L_ForceData'],L_ForceData],ignore_index=True)
                elif filename2.endswith('R_ForceData.csv'):
                    R_ForceData=pd.read_csv(filepath)
                    data['R_ForceData']=pd.concat([data['R_ForceData'],R_ForceData],ignore_index=True)'''
    return data

def load_matrix_data(first_dir= 'data/drawCircle')->dict: 
    data_dict=dict()
    timestamp_dict=dict()
    first_filenames=os.listdir(first_dir)
    data_dict['Pose']=None
    timestamp_dict['Pose']=None
    for filename1 in first_filenames:
        if filename1.endswith('__MACOSX'):
            continue
        else:
            second_dir=os.path.join(first_dir,filename1)
            second_filenames=os.listdir(second_dir)  
            is_episodes=True
            for filename2 in second_filenames:
                filepath=os.path.join(second_dir,filename2)
                if filename2.endswith('PoseData.csv'):
                    total_data=np.array(pd.read_csv(filepath,skiprows=1))#数据的第一行忽略，自己写coloums
                    Pose=np.array(total_data[:,1:])#转换为numpy
                    timestamp=np.array(total_data[:,0]) 
                    if data_dict['Pose'] is None or timestamp_dict['Pose'] is None:
                        data_dict['Pose']=Pose
                        timestamp_dict['Pose']=timestamp
                    else:
                        data_dict['Pose']=np.concatenate((data_dict['Pose'],Pose))# 注意这里的语法：将要连接的数组放在一个列表中 shape:(473, 16)
                        timestamp_dict['Pose']=np.concatenate((timestamp_dict['Pose'],timestamp)) 
    matrix=data_dict['Pose']
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
        # 转换为6维姿态向量 也许就是我想要的action or state
        pose = mat_to_pose(single_matrix)
        #print("6维姿态向量:", pose)
        if pose_matrix.size == 0:
            pose_matrix = pose
        else:
            pose_matrix=np.vstack((pose_matrix,pose))#vstack会自动处理维度,concatenate需要手动确保维度匹配             
    
    data_dict['Pose']=pose_matrix
    return data_dict,timestamp_dict



