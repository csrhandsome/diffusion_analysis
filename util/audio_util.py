import pandas as pd
import numpy as np
import seaborn as sns
from glob import glob
from time import time
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle
from scipy.signal import medfilt

def pad_audio(y, frame_length, hop_length):
    # 计算需要填充的长度
    pad_length = frame_length - len(y) % hop_length
    if pad_length == frame_length:
        pad_length = 0
    # 填充音频信号
    y_padded = np.pad(y, (0, pad_length), mode='constant')
    return y_padded

def read_audio(audio_file, frame_length=100, hop_length=100):
    try:
        # 读取音频
        y, sr = librosa.load(audio_file,sr=30)
        y, _ = librosa.effects.trim(y)
        y = np.pad(y, (0, 5000), mode='constant')
        
        # 计算时间戳
        timestamps = get_audio_timestamps(y, sr)
        
        # 确保同时返回音频数据和时间戳
        return y, timestamps
    except Exception as e:
        print(f"加载音频数据时出错: {str(e)}")
        return None, None

def read_audio_target_freq(audio_file, target_freq=30):
    """
    使用librosa的resample功能进行重采样
    """
    # 读取原始音频
    y, sr = librosa.load(audio_file)
    y, _ = librosa.effects.trim(y)
    
    # 计算目标采样点数
    target_length = int(len(y) * target_freq / sr)
    
    # 重采样
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_freq)
    
    # 补零
    y_resampled = np.pad(y_resampled, (0, 5000), mode='constant')
    
    return y_resampled, target_freq

def get_audio_timestamps(y, sr)->np.ndarray:
    duration = len(y) / sr  # 总时长（秒）
    # np.linspace(start, stop, num) 函数会生成 num 个等间距的点，从 start 到 stop
    # # 在这里 num=len(y)，所以生成的点数正好等于音频信号的采样点数
    timestamps = np.linspace(0, duration, len(y))
    return timestamps

def get_T1_T2(audio_file):
    IS = 0.25  # 静音段的长度，单位是秒
    wnd = 4096  # 帧大小
    inc = 400  # 帧移
    thr1 = 0.99  # 阈值1
    thr2 = 0.96  # 阈值2
    wlen = 4096  # 窗口长度
    y, sr = read_audio(audio_file, wnd, inc)
    NIS = int((IS * sr - wlen) // inc + 1)  # 计算静音段帧数
    # 将音频信号分帧，帧大小为 wnd，帧移为 inc，结果是一个二维数组，每行是一个帧
    frames = librosa.util.frame(y, frame_length=wnd, hop_length=inc).T
    # 对每一帧进行傅里叶变换
    frames = np.abs(np.fft.fft(frames, axis=1))
    # 计算频率分辨率
    df = sr / wlen
    fx1 = int(250 // df + 1)  # 250Hz 位置
    fx2 = int(3500 // df + 1)  # 3500Hz 位置
    km = wlen // 8
    K = 0.5  # 一个常数
    # 初始化能量矩阵
    E = np.zeros((frames.shape[0], wlen // 2))
    # 提取 250Hz 到 3500Hz 之间的频率分量
    E[:, fx1 + 1:fx2 - 1] = frames[:, fx1 + 1:fx2 - 1]
    # 将每个频率分量平方，计算能量
    E = np.multiply(E, E)
    # 计算每帧的总能量
    Esum = np.sum(E, axis=1, keepdims=True)
    # 计算能量分布比例
    P1 = np.divide(E, Esum)
    # 将能量分布比例大于等于 0.9 的频率分量置零
    E = np.where(P1 >= 0.9, 0, E)
    # 将频率分量分组，并计算每组的总能量
    Eb0 = E[:, 0::4]
    Eb1 = E[:, 1::4]
    Eb2 = E[:, 2::4]
    Eb3 = E[:, 3::4]
    Eb = Eb0 + Eb1 + Eb2 + Eb3
    # 计算每组的概率分布
    prob = np.divide(Eb + K, np.sum(Eb + K, axis=1, keepdims=True))
    # 计算每帧的熵
    Hb = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
    # 对熵进行平滑处理
    for i in range(10):
        Hb = medfilt(Hb, 5)
    # 计算平均熵
    Me = np.mean(Hb)
    # 计算静音段的平均熵
    eth = np.mean(Hb[:NIS])
    # 计算熵的差值
    Det = eth - Me
    # 计算阈值 T1 和 T2
    T1 = thr1 * Det + Me
    T2 = thr2 * Det + Me
    # 打印阈值和熵的形状
    print(f'T1 is {T1}, T2 is {T2}')
    #print(f'Hb is {Hb}')
    print(f'Hb shape is {Hb.shape}')
    # 调用 find_end_point 函数找到音频的端点
    SF, NF = find_end_point(Hb, T1, T2)
    return y, sr, Hb, SF, NF
     

def find_end_point(y, T1, T2):
    y_length = len(y)
    maxsilence = 8
    minlen = 5
    status = 0
    audio_length = np.zeros(y_length)
    audio_silence = np.zeros(y_length)
    segment_id = 0
    audio_start = np.zeros(y_length)
    audio_finish = np.zeros(y_length)
    for n in range(1, y_length):
        if status == 0 or status == 1:
            if y[n] < T2:
                audio_start[segment_id] = max(1, n - audio_length[segment_id] - 1)
                status = 2
                audio_silence[segment_id] = 0
                audio_length[segment_id] += 1
            elif y[n] < T1:
                status = 1
                audio_length[segment_id] += 1
            else:
                status = 0
                audio_length[segment_id] = 0
                audio_start[segment_id] = 0
                audio_finish[segment_id] = 0
        if status == 2:
            if y[n] < T1:
                audio_length[segment_id] += 1
            else:
                audio_silence[segment_id] += 1
                if audio_silence[segment_id] < maxsilence:
                    audio_length[segment_id] += 1
                elif audio_length[segment_id] < minlen:
                    status = 0
                    audio_silence[segment_id] = 0
                    audio_length[segment_id] = 0
                else:
                    status = 3
                    audio_finish[segment_id] = audio_start[segment_id] + audio_length[segment_id]
        if status == 3:
            status = 0
            segment_id += 1
            audio_length[segment_id] = 0
            audio_silence[segment_id] = 0
            audio_start[segment_id] = 0
            audio_finish[segment_id] = 0
    segment_num = len(audio_start[:segment_id])
    if audio_start[segment_num - 1] == 0:
        segment_num -= 1
    if audio_finish[segment_num - 1] == 0:
        print('Error: Not find ending point!\n')
        audio_finish[segment_num] = y_length
    SF = np.zeros(y_length)
    NF = np.ones(y_length)
    for i in range(segment_num):
        SF[int(audio_start[i]):int(audio_finish[i])] = 1
        NF[int(audio_start[i]):int(audio_finish[i])] = 0
    return SF, NF
