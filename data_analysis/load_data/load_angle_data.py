import os
import pandas as pd
import numpy as np
from util.pose_transform_util import *
# angle 其实反应的是action
# 0代表完全闭合，1代表完全张开，-1代表无数据
def load_angle_data(first_dir= 'data/drawCircle'): 
    data_dict=dict()
    timestamp_dict=dict()
    first_filenames=os.listdir(first_dir)
    data_dict['Angle']=None
    timestamp_dict['Angle']=None
    for filename1 in first_filenames:
        if filename1.endswith('__MACOSX'):
            continue
        else:
            second_dir=os.path.join(first_dir,filename1)
            second_filenames=os.listdir(second_dir)
            for filename2 in second_filenames:
                filepath=os.path.join(second_dir,filename2)
                if filename2.endswith('AngleData.csv'):
                    total_data=np.array(pd.read_csv(filepath,skiprows=1))
                    AngleData=np.array(total_data[:,1:])# 转换为numpy 不知道为什么total_data[:,1:]不行
                    timestamp=np.array(total_data[:,0]) 
                    if data_dict['Angle'] is None:
                        data_dict['Angle']=AngleData
                        timestamp_dict['Angle']=timestamp
                    else:
                        data_dict['Angle']=np.concatenate((data_dict['Angle'],AngleData))# 注意这里的语法：将要连接的数组放在一个列表中 shape:(473, 16)
                        timestamp_dict['Angle']=np.concatenate((timestamp_dict['Angle'],timestamp))
    return data_dict,timestamp_dict



