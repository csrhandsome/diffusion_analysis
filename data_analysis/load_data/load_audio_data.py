import os
import pandas as pd
import numpy as np
from util.audio_util import *

def load_audio_data(first_dir= 'data/drawCircle'): 
    data=dict()
    timestamp_dict=dict()
    first_filenames=os.listdir(first_dir)
    data['Audio']=None
    timestamp_dict['Audio']=None
    for filename1 in first_filenames:
        if filename1.endswith('__MACOSX'):
            continue
        else:
            second_dir=os.path.join(first_dir,filename1)
            second_filenames=os.listdir(second_dir)
            for filename2 in second_filenames:
                filepath=os.path.join(second_dir,filename2)
                if filename2.endswith('Audio.m4a'):
                    y, sr=read_audio_target_freq(filepath,target_freq=30)
                    AudioData=np.array(y)# 转换为numpy
                    timestamp=get_audio_timestamps(y,sr)# timestamp长度和AudioData长度相同
                    if data['Audio'] is None:
                        data['Audio']=AudioData
                        timestamp_dict['Audio']=timestamp
                    else:
                        data['Audio']=np.concatenate((data['Audio'],AudioData))
                        timestamp_dict['Audio']=np.concatenate((timestamp_dict['Audio'],timestamp))
    return data



