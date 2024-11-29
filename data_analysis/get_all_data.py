from data_analysis.load_data.load_angle_data import load_angle_data
from data_analysis.load_data.load_audio_data import load_audio_data
from data_analysis.load_data.load_depth_data import load_depth_data
from data_analysis.load_data.load_matrix_data import load_matrix_data
from data_analysis.load_data.load_video_data import load_video_data
from data_analysis.load_data.load_force_data import load_force_data
from data_analysis.align_data import simple_align
from data_analysis.load_data.load_episodes_data import load_episodes_data
import numpy as np

# data_dict 的键值有 Depth 、 Video 、 Angle 、 Pose 、 timestamp 、 episodes_ends 、Audio 、L_Force 、R_Force
def get_all_data(data_path='data/drawCircle'):
    print('------reading data begin-------')
    depth_data = load_depth_data()
    matrix_data,matrix_timestamp = load_matrix_data()
    video_data,video_timestamp= load_video_data()
    angle_data,angle_timestamp= load_angle_data()
    force_data,force_timestamp= load_force_data()
    #audio_data,audio_timestamp= load_audio_data()
    data_dict = {**matrix_data, **angle_data ,**video_data,**force_data,**depth_data}# 暂时不加depthdata
    timestamp_dict = {**matrix_timestamp, **angle_timestamp, **video_timestamp,**force_timestamp}
    '''for k,v in timestamp_dict.items():
        if k=='Depth':
            for i in range(len(v)):
                print(f'timestamp{k}[{i}] is {v[i]}')'''
    for k,v in data_dict.items():
        print(f'{k} shape: {v.shape}')
    data_dict= simple_align(data_dict,timestamp_dict)
    data_dict=load_episodes_data(data_dict)
    print('------after align-------')
    initial_state = []
    for k, v in data_dict.items():
        print(f'{k} shape: {v.shape}')
        if k in ['Depth', 'timestamp', 'episodes_ends']:
            continue
        # 确保v[0]是一维数组
        initial_value = np.array(v[0]).flatten()  # 将v[0]展平为一维
        initial_state.append(initial_value)
    # 使用np.concatenate拼接所有数组
    initial_state = np.concatenate(initial_state).reshape(1, -1)
    
    
    '''for i in range(len(data_dict['episodes_ends'])):
        print(f'data_dict["episodes_ends"][i] is {data_dict["episodes_ends"][i]}')'''
    print('------reading data end-------')
    return data_dict,initial_state


def test_videodict():
    video_data,video_timestamp= load_video_data()
    print(f'data["Video"] shape: {video_data["Video"].shape}')
    data_dict= simple_align(video_data,video_timestamp)
    load_episodes_data(data_dict)
    print('------after align-------')
    print(f'data["Video"] shape: {data_dict["Video"].shape}')
    # print(f'data["episodes_ends"] shape: {data["episodes_ends"].shape}')
    # for i in range(len(data['episodes_ends'])):
    # print(data['episodes_ends'][i])