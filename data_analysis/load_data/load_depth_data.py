import os
import pandas as pd
import numpy as np
from util.pose_transform_util import *
# 可以根据深度数据中的timestamp来截取视频中的frame吗，这样就可以实现一对一的对应
def load_depth_data(first_dir= 'data/drawCircle')->dict: 
    data_dict=dict()
    timestamp_dict=dict()
    first_filenames=os.listdir(first_dir)
    data_dict['Depth']=None
    timestamp_dict['Depth']=None
    for first_file in first_filenames:
        if first_file.endswith('__MACOSX'):
            continue
        else:
            second_dir=os.path.join(first_dir,first_file)
            second_filenames=os.listdir(second_dir)
            for second_file in second_filenames:
                if second_file.endswith('Depth'):
                    third_dir=os.path.join(second_dir,second_file)
                    third_finames=os.listdir(third_dir)
                    for third_file in third_finames:
                        if third_file.endswith('.bin'):
                            filepath=os.path.join(third_dir,third_file)
                            # 注意numpy是行优先，所以是(height, width) 实际上是256*192
                            DepthData=np.fromfile(filepath,dtype=np.uint16).reshape(192, 256) /10000
                            # filepath: data/drawCircle\20240912_210544_Unspecified\Depth\depth_0.000.bin
                            part=filepath.split('_')[-1].split('.')[:2]
                            #timestamp=np.array([float(part[0]+'.'+part[1])])
                            if data_dict['Depth'] is None:
                                data_dict['Depth']=DepthData
                                #timestamp_dict['Depth']=timestamp
                            else:
                                data_dict['Depth']=np.concatenate((data_dict['Depth'],DepthData))# 注意这里的语法：将要连接的数组放在一个列表中 shape:(473, 16)
                                #timestamp_dict['Depth']=np.append(timestamp_dict['Depth'], timestamp)
    data_dict['Depth']=data_dict['Depth'].reshape(-1, 192, 256) # (batch, height, width)
    return data_dict



