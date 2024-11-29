import os
import pandas as pd
import numpy as np
from util.pose_transform_util import *
def load_force_data(first_dir= 'data/drawCircle'): 
    data_dict=dict()
    timestamp_dict=dict()
    first_filenames=os.listdir(first_dir)
    data_dict['L_Force']=None
    timestamp_dict['L_Force']=None
    data_dict['R_Force']=None
    timestamp_dict['R_Force']=None
    for filename1 in first_filenames:
        if filename1.endswith('__MACOSX'):
            continue
        else:
            second_dir=os.path.join(first_dir,filename1)
            second_filenames=os.listdir(second_dir)
            for filename2 in second_filenames:
                filepath=os.path.join(second_dir,filename2)
                if filename2.endswith('L_ForceData.csv'):
                    total_data=np.array(pd.read_csv(filepath,skiprows=1))
                    L_ForeceData=np.array(total_data[:,1:])
                    timestamp=np.array(total_data[:,0]) 
                    if data_dict['L_Force'] is None:
                        data_dict['L_Force']=L_ForeceData
                        timestamp_dict['L_Force']=timestamp
                    else:
                        data_dict['L_Force']=np.concatenate((data_dict['L_Force'],L_ForeceData))# 注意这里的语法：将要连接的数组放在一个列表中 shape:(473, 16)
                        timestamp_dict['L_Force']=np.concatenate((timestamp_dict['L_Force'],timestamp))
                if filename2.endswith('R_ForceData.csv'):
                    total_data=np.array(pd.read_csv(filepath,skiprows=1))
                    R_ForeceData=np.array(total_data[:,1:])
                    timestamp=np.array(total_data[:,0]) 
                    if data_dict['R_Force'] is None:
                        data_dict['R_Force']=R_ForeceData
                        timestamp_dict['R_Force']=timestamp
                    else:
                        data_dict['R_Force']=np.concatenate((data_dict['R_Force'],R_ForeceData))# 注意这里的语法：将要连接的数组放在一个列表中 shape:(473, 16)
                        timestamp_dict['R_Force']=np.concatenate((timestamp_dict['R_Force'],timestamp))
    return data_dict,timestamp_dict



