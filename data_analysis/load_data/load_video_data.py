import os
import cv2
from util.pose_transform_util import *
from torchvision import transforms
from diffusion.model.visionencoder import *
from data.global_data import *
from pathlib import Path
from data_analysis.load_data.load_depth_data import *


def load_video_data(first_dir= 'data/drawCircle'): 
    # 加载resnet18模型
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)# return nn.Module 要输入numpy.array
    data=dict()
    timestamp_dict=dict()
    data['Video']=None
    timestamp_dict['Video']=None
    first_filenames=os.listdir(first_dir)
    for filename1 in first_filenames:
        if filename1.endswith('__MACOSX'):
            continue
        else:
            second_dir=os.path.join(first_dir,filename1)
            second_filenames=os.listdir(second_dir)
            for filename2 in second_filenames:
                filepath=os.path.join(second_dir,filename2)
                if filename2.endswith('RGB.mp4'):
                    frames,timestamp=video_to_tensor(filepath)# shape: [1, num_frames, channels, height, width] torch.tensor]
                    frames=frames.reshape(frames.shape[0]*frames.shape[1],3,256,192)
                    feature=vision_encoder(frames)# feature shape: torch.Size([num_frames, 512])
                    feature=feature.detach().numpy()# 后续要作为model的globalcond,所以要转为numpy.array,若不转则使用torch.cat
                    if data['Video'] is None:
                        data['Video']=feature
                        timestamp_dict['Video']=timestamp
                    else:
                        #data['Video']=torch.cat((data['Video'],feature),dim=0)
                        data['Video'] = np.concatenate((data['Video'], feature), axis=0)
                        timestamp_dict['Video']=np.concatenate((timestamp_dict['Video'],timestamp))
    return data,timestamp_dict


def select_cap(video_path):
    backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    
    for backend in backends:
        cap = cv2.VideoCapture(video_path, backend)
        if cap.isOpened():
            print(f"Successfully opened with backend {backend}")
            return cap
    
    raise Exception("Could not open video with any backend")


def video_to_tensor(video_path, num_frames=10, height=256, width=192):
    video_path = str(Path(video_path))# 确保路径无误
    # 打开视频文件
    #根据select_cap函数选出的backend
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Failed to open video with CAP_FFMPEG")
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    #print(f"Total frames: {total_frames}")
    # 计算采样间隔
    step = 1
    num_frames = total_frames
    # 预处理转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),  # 将PIL图像转换为tensor，并归一化到[0,1]
    ])
    frames = []
    timestamps = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0 and len(frames) < num_frames:
            # OpenCV读取的是BGR格式，转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为tensor并添加到列表
            frame_tensor = transform(frame)
            frames.append(frame_tensor)
            # 计算时间戳
            timestamp = frame_count/fps
            timestamps.append(timestamp)
        frame_count += 1
        if len(frames) == num_frames:
            break
    cap.release()
    # 堆叠所有帧
    frames = torch.stack(frames)  # shape: [num_frames, channels, height, width]
    # 添加batch维度
    frames = frames.unsqueeze(0)  # shape: [1, num_frames, channels, height, width]
    timestamps=np.array(timestamps)
    return frames,timestamps
