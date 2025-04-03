import h5py
import zarr
from pathlib import Path
import numpy as np


def halfdownsample(data):
    i = 0
    j = 0
    new_data = np.zeros((data.shape[0] // 2, *data.shape[1:]), dtype=data.dtype)
    while i < new_data.shape[0]:
        new_data[i] = data[j]
        i += 1
        j += 2
    return new_data


dir = Path("data/put_doll_in_box")
front_images = list()
wrist_images = list()
low_dims = list()
actions = list()
episode_lens = list()
for episode in list(dir.glob("*.hdf5")):
    with h5py.File(episode, 'r') as f:
        front_image = f['/front_camera_images'][()]
        wrist_image = f['/wrist_camera_images'][()]
        action = f['/actions'][()]
        # 转为相对运动   
        # action[1:, :6] - action[:-1, :6]计算相邻时间步动作的前6个维度之间的差值
        action[:, :6] = np.concatenate((action[1:, :6] - action[:-1, :6], np.zeros((1, 6), dtype=action.dtype)))
        low_dim = f['/low_dims'][()]# state

        if not front_image.shape[0] == wrist_image.shape[0] == low_dim.shape[0] == action.shape[0]:
            print("not aligned: ", str(episode))
        if action.shape[0] > 24 * 9:
            print("exceed: ", str(episode), action.shape[0])

        front_images.append(front_image)
        wrist_images.append(wrist_image)
        low_dims.append(low_dim)
        actions.append(action)
        episode_lens.append(action.shape[0])
        f.close()

front_images = np.concatenate(front_images)    
wrist_images = np.concatenate(wrist_images) 
low_dims = np.concatenate(low_dims)
actions = np.concatenate(actions)
episode_lens = np.cumsum(episode_lens)
with zarr.open("train_data.zarr", 'w') as f:
    f['/data/front_camera_images'] = front_images
    f['/data/wrist_camera_images'] = wrist_images
    f['/data/low_dims'] = low_dims
    f['/data/actions'] = actions
    f['/meta/episode_ends'] = episode_lens



# with zarr.open("train1.zarr", 'r') as f:
#     front_images1 = f['/data/front_camera_images'][()]
#     wrist_images1 = f['/data/wrist_camera_images'][()]
#     low_dims1 = f['/data/low_dims'][()]
#     actions1 = f['/data/actions'][()]
#     episode_lens1 = f['/meta/episode_ends'][()]

# with zarr.open("train2.zarr", 'r') as f:
#     front_images2 = f['/data/front_camera_images'][()]
#     wrist_images2 = f['/data/wrist_camera_images'][()]
#     low_dims2 = f['/data/low_dims'][()]
#     actions2 = f['/data/actions'][()]
#     episode_lens2 = f['/meta/episode_ends'][()]

# front_images = np.concatenate([front_images1, front_images2])
# wrist_images = np.concatenate([wrist_images1, wrist_images2])
# low_dims = np.concatenate([low_dims1, low_dims2])
# actions = np.concatenate([actions1, actions2])
# episode_lens = np.concatenate([episode_lens1, episode_lens2])

# with zarr.open("train.zarr", 'w') as f:
#     f['/data/front_camera_images'] = front_images
#     f['/data/wrist_camera_images'] = wrist_images
#     f['/data/low_dims'] = low_dims
#     f['/data/actions'] = actions
#     f['/meta/episode_ends'] = episode_lens