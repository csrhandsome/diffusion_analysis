import os
import pandas as pd
import numpy as np
import fnmatch
def load_force_data(first_dir= 'data/drawCircle'): 
    L_Force=None
    R_Force=None
    for root, dirs, files in os.walk(first_dir):
        if "__MACOSX" in dirs:
            dirs.remove("__MACOSX") # 跳过__MACOSX目录
        for file in files:
            if file == 'L_ForceData.csv':
                file_path = os.path.join(root, file)
                total_data=np.array(pd.read_csv(file_path,skiprows=1))
                L_ForeceData=np.array(total_data[:,1:])
                if L_Force is None:
                    L_Force=L_ForeceData
                else:
                    L_Force=np.concatenate((L_Force,L_ForeceData))# 注意这里的语法：将要连接的数组放在一个列表中 shape:(473, 16)
            if file == 'R_ForceData.csv':
                file_path = os.path.join(root, file)
                total_data=np.array(pd.read_csv(file_path,skiprows=1))
                R_ForeceData=np.array(total_data[:,1:])
                if R_Force is None:
                    R_Force=R_ForeceData
                else:
                    R_Force=np.concatenate((R_Force,R_ForeceData))# 注意这里的语法：将要连接的数组放在一个列表中 shape:(473, 16)
    return L_Force,R_Force

def upsample_force_data(L_Force,R_Force,upsample_factor=4):
    L_Force_upsampled = np.zeros((L_Force.shape[0] * upsample_factor, L_Force.shape[1]))
    R_Force_upsampled = np.zeros((R_Force.shape[0] * upsample_factor, R_Force.shape[1]))
    for i in range(L_Force.shape[1]):
        L_Force_upsampled[:, i] = np.interp(np.arange(L_Force.shape[0] * upsample_factor), np.arange(L_Force.shape[0]), L_Force[:, i])
        R_Force_upsampled[:, i] = np.interp(np.arange(R_Force.shape[0] * upsample_factor), np.arange(R_Force.shape[0]), R_Force[:, i])
    return L_Force_upsampled,R_Force_upsampled

 