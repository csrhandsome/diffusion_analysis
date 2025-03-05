import os
import cv2
import torch
import fnmatch
import numpy as np
from torchvision import transforms
from diffusion.model.vision.resnet_visionencoder import get_resnet,replace_bn_with_gn
from data.global_data import *
from pathlib import Path
from data_analysis.load_data.load_depth_data import load_depth_data


def load_video_data(first_dir= 'data/drawCircle',weights=None,return_feature=True): 
    # 加载resnet18模型
    vision_encoder = get_resnet('resnet18',weights=weights)# weights='r3m' 要求输入为(C, H, W)
    vision_encoder = replace_bn_with_gn(vision_encoder)# return nn.Module 要输入numpy.array
    Video=None
    for root, dirs, files in os.walk(first_dir):
        if "__MACOSX" in dirs:
            dirs.remove("__MACOSX") # 跳过__MACOSX目录
        for file in files:
            if file == 'RGB.mp4':
                file_path = os.path.join(root, file)
                frames = video_to_frames_new(file_path) # 形状:[num_frames,256,192,3]
                frames = frames.permute(0,3,1,2)# 改为resnet需要的形状 [num_frames,3,256,192]，该行可注释
                if return_feature:
                    frames = vision_encoder(frames) # feature shape: torch.Size([num_frames, 512])
                    frames = frames.detach().numpy() # 后续要作为model的globalcond,所以要转为numpy.array,若不转则使用torch.cat
                if Video is None:
                    Video = frames
                else:
                    #data['Video']=torch.cat((data['Video'],feature),dim=0)
                    Video = np.concatenate((Video, frames), axis=0)
    return Video


def select_cap(video_path):
    backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    
    for backend in backends:
        cap = cv2.VideoCapture(video_path, backend)
        if cap.isOpened():
            print(f"Successfully opened with backend {backend}")
            return cap
    
    raise Exception("Could not open video with any backend")


def video_to_frame(video_path, num_frames=10, height=256, width=192)-> torch.Tensor:
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

def video_to_frames_new(video_path, num_frames=10, height=256, width=192)-> torch.Tensor:
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
    frames = torch.stack(frames)# shape: [num_frames, channels, height, width]
    # 添加batch维度
    frames = frames.permute(0, 2, 3, 1)# 形状:[num_frames,256,192,3] shape:[num_frames,height,width,channels]
    return frames
