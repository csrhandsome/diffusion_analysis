from data_analysis.load_data.load_angle_data import load_angle_data
from data_analysis.load_data.load_audio_data import load_audio_data
from data_analysis.load_data.load_depth_data import load_depth_data
from data_analysis.load_data.load_gripper_pose_data import load_gripper_pose_data
from data_analysis.load_data.load_video_data import load_video_data
from data_analysis.load_data.load_force_data import load_force_data,upsample_force_data
from data_analysis.load_data.load_episodes_data import load_episodes_length
from data_analysis.load_data.load_qpos_data import load_qpos_data
from data_analysis.align_data import simple_align
import os
import numpy as np
import h5py
# data_dict 的键值有 Depth 、 Video 、 Angle 、 Pose 、 timestamp 、 episodes_ends 、Audio 、L_Force 、R_Force
def get_all_data(data_path='data/drawCircle',specific_episode=None):
    ''' 采集数据和对齐帧率 '''
    # print('------reading data begin-------')
    # 计算episodes
    episode_ends=[0]
    start_idx = 0
    episode_lens = load_episodes_length(data_path)
    for i in range(len(episode_lens)):
        start_idx += episode_lens[i]
        episode_ends.append(start_idx)
    # 如果指定了特定的 episode，则构建完整路径
    if specific_episode:
        episode_path = os.path.join(data_path, specific_episode)
        if not os.path.exists(episode_path):
            print(f"错误：指定的 episode 路径 {episode_path} 不存在")
            return None
        data_path = episode_path
    depth_data = load_depth_data(data_path)
    gripper_pose_data,episode_lens = load_gripper_pose_data(data_path)
    video_data = load_video_data(data_path,return_feature=False) # return_feature=True返回(n,512)
    angle_data = load_angle_data(data_path)
    #audio_data = load_audio_data(data_path,target_freq=29.75)
    L_Force_data,R_Force_data = load_force_data(data_path)# 力的数据为15HZ，因此需要将所有的数据降频为15HZ
    qpos_data = load_qpos_data(data_path)
    
    '''# 方案一 ：采样降频 这些数据原本是60HZ 降到15HZ
    depth_data = depth_data[::4]
    gripper_pose_data = gripper_pose_data[::4]
    video_data = video_data[::4]
    angle_data = angle_data[::4]
    audio_data = audio_data[::4]
    # 方案一当中更新episode_ends,因为帧数减为1/4了
    episode_ends = [end // 4 for end in episode_ends]'''
    
    # 方案二：将力的数据增频到60HZ（上采样） 上采用极其难对齐
    original_sr = 15  
    target_sr = 60   
    upsample_factor = int(target_sr / original_sr)
    L_Force_data,R_Force_data=upsample_force_data(L_Force_data,R_Force_data,upsample_factor)
    

    all_data = [depth_data, gripper_pose_data, video_data, angle_data, L_Force_data, R_Force_data]
    '''for i in range(len(all_data)):
        print(f'all_data[{i}] shape is {all_data[i].shape}')'''
    aligned_list = simple_align(all_data)
    depth_data, gripper_pose_data, video_data, angle_data, L_Force_data, R_Force_data= aligned_list
    '''print('------after align-------')
    for i in range(len(aligned_list)):
        print(f'aligned_list[{i}] shape is {aligned_list[i].shape}')'''

    
    # print('------reading data end-------')
    return {
            "observations": {
                "qpos": np.concatenate([gripper_pose_data, angle_data], axis=1),# 机械臂上的传感器读取执行后的实际状态（qpos）目前没有
                "images": {
                    "cam_high": video_data,
                    "cam_phone": video_data,
                },
                "phone_cam_depth": depth_data,
            },
            "L_Force": L_Force_data,
            "R_Force": R_Force_data,
            #"audio": audio_data,
            "actions": np.concatenate([gripper_pose_data, angle_data], axis=1),# 夹爪的数据为action
            "griper_angle": angle_data,
            "episodes_ends": episode_ends,
            "episode_lens": episode_lens
            }

def get_all_episode_paths(data_path='data/drawCircle'):
    '''
    获取指定数据路径下的所有 episode 目录
    '''
    episode_paths = []
    
    # 遍历数据目录
    for root, dirs, files in os.walk(data_path):
        if "__MACOSX" in dirs:
            dirs.remove("__MACOSX")  # 跳过__MACOSX目录
        
        # 检查当前目录是否包含 PoseData.csv 文件
        if 'PoseData.csv' in files:
            # 返回的路径是相对于 data/drawCircle 的相对路径 ,也就是目录名
            rel_path = os.path.relpath(root, data_path)
            if rel_path != '.':  # 不是根目录
                episode_paths.append(rel_path)
    
    return episode_paths

def get_all_data_to_hdf5(data_path='data/drawCircle',specific_episode=None):
    ''' 采集数据和对齐帧率 '''
    # print('------reading data begin-------')
    # 计算episodes
    episode_ends=[0]
    start_idx = 0
    episode_lens = load_episodes_length(data_path)
    for i in range(len(episode_lens)):
        start_idx += episode_lens[i]
        episode_ends.append(start_idx)
    # 如果指定了特定的 episode，则构建完整路径
    if specific_episode:
        episode_path = os.path.join(data_path, specific_episode)
        if not os.path.exists(episode_path):
            print(f"错误：指定的 episode 路径 {episode_path} 不存在")
            return None
        data_path = episode_path
    depth_data = load_depth_data(data_path)
    gripper_pose_data,episode_lens = load_gripper_pose_data(data_path)
    video_data = load_video_data(data_path,return_feature=False) # return_feature=True返回(n,512)
    angle_data = load_angle_data(data_path)
    #audio_data = load_audio_data(data_path,target_freq=29.75)
    L_Force_data,R_Force_data = load_force_data(data_path)# 力的数据为15HZ，因此需要将所有的数据降频为15HZ
    qpos_data = load_qpos_data(data_path)
    
    '''# 方案一 ：采样降频 这些数据原本是60HZ 降到15HZ
    depth_data = depth_data[::4]
    gripper_pose_data = gripper_pose_data[::4]
    video_data = video_data[::4]
    angle_data = angle_data[::4]
    audio_data = audio_data[::4]
    # 方案一当中更新episode_ends,因为帧数减为1/4了
    episode_ends = [end // 4 for end in episode_ends]'''
    
    # 方案二：将力的数据增频到60HZ（上采样） 上采用极其难对齐
    original_sr = 15  
    target_sr = 60   
    upsample_factor = int(target_sr / original_sr)
    L_Force_data,R_Force_data=upsample_force_data(L_Force_data,R_Force_data,upsample_factor)
    

    all_data = [depth_data, gripper_pose_data, video_data, angle_data, L_Force_data, R_Force_data]
    '''for i in range(len(all_data)):
        print(f'all_data[{i}] shape is {all_data[i].shape}')'''
    aligned_list = simple_align(all_data)
    depth_data, gripper_pose_data, video_data, angle_data, L_Force_data, R_Force_data= aligned_list
    '''print('------after align-------')
    for i in range(len(aligned_list)):
        print(f'aligned_list[{i}] shape is {aligned_list[i].shape}')'''

    # print('------reading data end-------')
    result_dict = {
            "observations": {
                "qpos": np.concatenate([gripper_pose_data, angle_data], axis=1),# 机械臂上的传感器读取执行后的实际状态（qpos）目前没有
                "images": {
                    "cam_high": video_data,
                    "cam_phone": video_data,
                },
                "phone_cam_depth": depth_data,
            },
            "L_Force": L_Force_data,
            "R_Force": R_Force_data,
            #"audio": audio_data,
            "actions": np.concatenate([gripper_pose_data, angle_data], axis=1),# 夹爪的数据为action
            "griper_angle": angle_data,
            "episodes_ends": episode_ends,
            "episode_lens": episode_lens
            }
    
    current_dir = os.path.basename(data_path)
    parent_dir = os.path.dirname(data_path)
    hdf5_path = os.path.join(parent_dir, f'{current_dir}.hdf5')
    print(f'hdf5_path is {hdf5_path}')
    # 将数据保存为HDF5格式
    with h5py.File(hdf5_path, 'w') as f:
        # 保存observations组
        obs_group = f.create_group('observations')
        obs_group.create_dataset('qpos', data=result_dict['observations']['qpos'])
        
        # 创建images子组
        images_group = obs_group.create_group('images')
        images_group.create_dataset('cam_high', data=result_dict['observations']['images']['cam_high'])
        images_group.create_dataset('cam_phone', data=result_dict['observations']['images']['cam_phone'])  
        obs_group.create_dataset('phone_cam_depth', data=result_dict['observations']['phone_cam_depth'])
        
        # 保存其他数据
        f.create_dataset('L_Force', data=result_dict['L_Force'])
        f.create_dataset('R_Force', data=result_dict['R_Force'])
        f.create_dataset('actions', data=result_dict['actions'])
        f.create_dataset('griper_angle', data=result_dict['griper_angle'])
        f.create_dataset('episodes_ends', data=result_dict['episodes_ends'])
        f.create_dataset('episode_lens', data=result_dict['episode_lens'])

    return result_dict

def convert_all_to_hdf5(data_path='data/drawCircle'):
    """
    将指定目录下的所有episode数据转换为HDF5格式
    """
    print(f'开始将{data_path}下的所有episode数据转换为HDF5格式...')
    # 获取所有episode目录
    episode_paths = get_all_episode_paths(data_path)
    successful_hdf5_files = []
    
    # 遍历每个episode并转换
    for i, episode_path in enumerate(episode_paths):
        print(f'正在处理 [{i+1}/{len(episode_paths)}]: {episode_path}')
        try:
            # 构建完整路径
            full_episode_path = os.path.join(data_path, episode_path)
            # 检查对应的HDF5文件是否已存在
            hdf5_file_path = os.path.join(data_path, f'{episode_path}.hdf5')
            if os.path.exists(hdf5_file_path):
                print(f'HDF5文件已存在: {hdf5_file_path}，跳过转换')
                successful_hdf5_files.append(hdf5_file_path)
                continue
            # 转换数据到HDF5格式
            result_dict = get_all_data_to_hdf5(full_episode_path)
            successful_hdf5_files.append(os.path.join(data_path, f'{episode_path}.hdf5'))
            print(f'成功转换: {episode_path}')
        except Exception as e:
            print(f'转换失败 {episode_path}: {str(e)}')
    
    print(f'转换完成! 共成功转换 {len(successful_hdf5_files)}/{len(episode_paths)} 个episode')
    return successful_hdf5_files