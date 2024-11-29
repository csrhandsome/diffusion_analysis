import os
import zarr
import numpy as np

data_sum=25650
data_path='pusht_cchi_v7_replay.zarr.zip'

def random_range_step(length,min_val,max_val,dimension,data):#把维度2的数据改成在（130，300）区间的两个数据，并且在其中不断的递增，达到上限后再不断的递减，如此反复
    current_val=min_val
    increasing=True
    for i in range(length):
        if dimension==2:
            step = np.random.randint(6, 11)  # 随机生成1到4之间的整数作为步长
        else:
            step=np.random.randint(5,10)
        for j in range(dimension):
            data[i, j] = current_val
            if increasing:
                current_val += step
                if current_val >= max_val:
                    increasing = False
                    current_val = max_val
            else:
                current_val -= step
                if current_val <= min_val:
                    increasing = True
                    current_val = min_val
    return data

def random_input_for_dataloader() :
    
    if os.path.exists('data.zarr.zip'):
        print('data.zip exists')
        return
    store=zarr.ZipStore('data.zarr.zip')
    root=zarr.group(store=store)
    img=np.random.randn(data_sum,96,96,3)
    keypoint=np.random.randn(data_sum,9,2)
    n_contacts=np.random.randn(data_sum,1)

    # 生成在 130 到 300 之间递增和递减的数据，步长为1-4之间的随机值
    action = np.zeros((data_sum, 2))
    action=random_range_step(data_sum,130,449,2,action)
    state = np.zeros((data_sum, 5))
    state=random_range_step(data_sum,130,449,5,state)
    
    
    episode_ends=np.zeros(data_sum,dtype=np.int64)
    current_value=9
    for i in range(206):
        step=np.random.randint(10,20)
        current_value+=step
        episode_ends[i]=current_value
    
    for i in range(len(action)):
        #np.all()check if there all the elements in the array are one
        while np.all(action[i]==0) or np.all(action[i]==1):
            action[i]=np.random.randn(2)
            state[i]=np.random.randn(5)

    root.create_dataset('img',data=img)
    root.create_dataset('keypoint',data=keypoint)
    root.create_dataset('n_contacts',data=n_contacts)
    root.create_dataset('action',data=action)
    root.create_dataset('state',data=state)
    root.create_dataset('episode_ends',data=episode_ends)
    store.close()
    print(f'data.zip created')