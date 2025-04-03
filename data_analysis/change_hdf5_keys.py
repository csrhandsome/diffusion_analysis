import h5py
from pathlib import Path
import shutil  # 用于备份原文件

def convert_hdf5_to_nested_structure(original_path, new_path):
    """转换HDF5文件结构，使其支持 f['images']['cam_high'] 访问方式"""
    with h5py.File(original_path, 'r') as f_orig, h5py.File(new_path, 'w') as f_new:
        # 1. 创建 'images' 组，并添加相机数据
        images_group = f_new.create_group('images')
        
        if '/front_camera_images' in f_orig:
            images_group['cam_high'] = f_orig['/front_camera_images'][()]
        
        if '/wrist_camera_images' in f_orig:
            images_group['cam_phone'] = f_orig['/wrist_camera_images'][()]

        # 2. 添加动作数据（action）
        if '/actions' in f_orig:
            f_new['action'] = f_orig['/actions'][()]  # 注意：单数形式

        # 3. 添加状态数据（state）
        if '/low_dims' in f_orig:
            f_new['state'] = f_orig['/low_dims'][()]

        # 4. 复制其他数据（如果有）
        for key in f_orig.keys():
            if key not in ['front_camera_images', 'wrist_camera_images', 'actions', 'low_dims']:
                f_new[key] = f_orig[key][()]

def batch_convert_hdf5(input_dir, output_dir):
    """批量转换目录下的所有HDF5文件"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for hdf5_file in input_dir.glob("*.hdf5"):
        new_file = output_dir / hdf5_file.name
        convert_hdf5_to_nested_structure(hdf5_file, new_file)
        print(f"Converted: {hdf5_file} -> {new_file}")

# 使用方法：
batch_convert_hdf5("data/put_doll_in_box", "data/put_doll_in_box_new")