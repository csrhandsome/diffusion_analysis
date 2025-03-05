import os
import numpy as np
import fnmatch
from util.audio_util import read_audio_target_freq,get_audio_timestamps

def load_audio_data(first_dir= 'data/drawCircle',target_freq=29): 
    Audio=None
    for root, dirs, files in os.walk(first_dir):
        if "__MACOSX" in dirs:
            dirs.remove("__MACOSX") # 跳过__MACOSX目录
        for file in files:
            if file == 'Audio.m4a':
                    file_path = os.path.join(root, file)
                    y, sr=read_audio_target_freq(file_path,target_freq=target_freq)
                    AudioData=np.array(y)# 转换为numpy
                    # timestamp=get_audio_timestamps(y,sr)# timestamp长度和AudioData长度相同
                    if Audio is None:
                        Audio = AudioData
                    else:
                        Audio = np.concatenate((Audio,AudioData))
    return Audio



