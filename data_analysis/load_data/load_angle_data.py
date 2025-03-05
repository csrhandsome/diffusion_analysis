import os
import pandas as pd
import numpy as np
import fnmatch
# angle 其实反应的是action
# 0代表完全闭合，1代表完全张开，-1代表无数据
def load_angle_data(first_dir= 'data/drawCircle'): 
    Angle=None
    for root, dirs, files in os.walk(first_dir):
        if "__MACOSX" in dirs:
            dirs.remove("__MACOSX") # 跳过__MACOSX目录
        for file in files:
            if file == 'AngleData.csv':
                    file_path = os.path.join(root, file)
                    total_data=np.array(pd.read_csv(file_path,skiprows=1))
                    AngleData=np.array(total_data[:,1:])# 转换为numpy 不知道为什么total_data[:,1:]不行
                    if Angle is None:
                        Angle = AngleData
                    else:
                        Angle = np.concatenate((Angle,AngleData))# 注意这里的语法：将要连接的数组放在一个列表中 shape:(473, 16)
    return Angle



