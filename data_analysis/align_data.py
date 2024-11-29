import numpy as np
from scipy.interpolate import interp1d
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def interpolate_array(data, source_timestamps, target_timestamps):
    """
    递归处理任意维度数组的插值。
    
    Args:
        data (np.ndarray): 要插值的数据，任意维度。
        source_timestamps (np.ndarray): 源数据的时间戳。
        target_timestamps (np.ndarray): 目标插值时间戳。
    
    Returns:
        np.ndarray: 插值后的数据。
    """
    if len(data.shape) == 1:
        # 一维数据，直接插值
        interp_func = interp1d(source_timestamps, data, kind='linear', bounds_error=False, fill_value="extrapolate")
        return interp_func(target_timestamps)
    else:
        # 多维数据，对每一维递归插值
        result = np.zeros((len(target_timestamps),) + data.shape[1:])
        for i in range(data.shape[1]):
            result[:, i] = interpolate_array(data[:, i], source_timestamps, target_timestamps)
        return result

def find_segments(timestamps, gap_threshold=1.0):
    """
    找出时间戳的分段。
    
    Args:
        timestamps (np.ndarray): 时间戳数组。
        gap_threshold (float): 判定为新段的时间间隔阈值（秒）。
    
    Returns:
        list of tuples: 每段的起始和结束索引列表 [(start1, end1), (start2, end2), ...]。
    """
    segments = []
    start_idx = 0
    
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i-1] > gap_threshold:
            segments.append((start_idx, i))
            start_idx = i
    
    segments.append((start_idx, len(timestamps)))
    return segments

def align_multi_modal_data_by_shortest_segments(data_dict, timestamps_dict, gap_threshold=1.0):
    """
    按照更短的段数对齐多模态数据，保持每段的长度一致，确保所有模态的数据长度相同。
    
    Args:
        data_dict (dict): 数据字典 {key1: array1, key2: array2, ...}。
        timestamps_dict (dict): 时间戳字典 {key1: array1, key2: array2, ...}。
        gap_threshold (float): 判定为新段的时间间隔阈值（秒）。
    
    Returns:
        tuple:
            aligned_data (dict): 对齐后的数据字典 {key1: aligned_array1, key2: aligned_array2, ...}。
            aligned_timestamps (list of np.ndarray): 对齐后的每段统一的时间戳列表。
    """
    # 基本验证
    if not data_dict or not timestamps_dict:
        raise ValueError("数据字典和时间戳字典不能为空")
    
    # 转换时间戳为numpy数组并确保排序
    timestamps_dict = {k: np.array(v, dtype=np.float64) for k, v in timestamps_dict.items()}
    for k in timestamps_dict:
        if not np.all(np.diff(timestamps_dict[k]) >= 0):
            raise ValueError(f"时间戳数组 {k} 必须是递增的")
    
    # 找出每个模态的数据段
    segments_dict = {k: find_segments(ts, gap_threshold) for k, ts in timestamps_dict.items()}
    logger.info(f"Segments per modal: {segments_dict}")
    
    # 计算所有模态的段数，选择最少段数来进行对齐
    min_num_segments = min(len(v) for v in segments_dict.values())
    logger.info(f"Minimum number of segments to align: {min_num_segments}")
    
    # 初始化对齐后的数据结构
    aligned_data = {k: [] for k in data_dict.keys()}
    aligned_timestamps = []
    
    # 对每个段进行对齐
    for seg_idx in range(min_num_segments):
        # 收集当前段各模态的起始和结束时间
        current_segment_times = []
        for k in data_dict.keys():
            start, end = segments_dict[k][seg_idx]
            current_segment_times.append((timestamps_dict[k][start], timestamps_dict[k][end-1]))
        
        # 确定当前段的统一起始和结束时间
        master_start = max(seg[0] for seg in current_segment_times)
        master_end = min(seg[1] for seg in current_segment_times)
        
        if master_end <= master_start:
            logger.warning(f"Segment {seg_idx} has no overlapping time range. Skipping.")
            continue
        
        # 对每个模态，在当前段内提取数据和时间戳，并进行截取到重叠时间范围
        segment_data_dict = {}
        for k in data_dict.keys():
            start, end = segments_dict[k][seg_idx]
            segment_data = data_dict[k][start:end]
            segment_timestamps = timestamps_dict[k][start:end]
            
            # 找到重叠时间范围内的数据
            mask = (segment_timestamps >= master_start) & (segment_timestamps <= master_end)
            trimmed_timestamps = segment_timestamps[mask]
            trimmed_data = segment_data[mask]
            
            # 如果某个模态在重叠时间范围内没有数据，则跳过当前段
            if len(trimmed_timestamps) < 2:
                logger.warning(f"Modal {k} segment {seg_idx} has insufficient data after trimming. Skipping segment.")
                break
            
            segment_data_dict[k] = (trimmed_data, trimmed_timestamps)
        
        else:
            # 如果所有模态在重叠时间范围内都有足够的数据
            # 确定每个模态在当前段的点数
            num_points_per_modal = {k: len(v[1]) for k, v in segment_data_dict.items()}
            
            # 选择该段所有模态中最小的点数作为目标点数
            target_num_points = min(num_points_per_modal.values())
            logger.info(f"Segment {seg_idx}: target_num_points={target_num_points}")
            
            # 定义统一的目标时间戳
            target_timestamps = np.linspace(master_start, master_end, target_num_points)
            aligned_timestamps.append(target_timestamps)
            
            # 对每个模态进行插值，以匹配统一的目标时间戳
            for k in data_dict.keys():
                segment_data, segment_timestamps = segment_data_dict[k]
                interpolated_data = interpolate_array(segment_data, segment_timestamps, target_timestamps)
                aligned_data[k].append(interpolated_data)
    
    # 合并每个模态的所有段数据
    for k in aligned_data.keys():
        if aligned_data[k]:
            aligned_data[k] = np.concatenate(aligned_data[k], axis=0)
        else:
            aligned_data[k] = np.array([])  # 如果没有对齐的数据段，返回空数组
    
    # 合并所有段的时间戳
    if aligned_timestamps:
        aligned_timestamps = np.concatenate(aligned_timestamps)
    else:
        aligned_timestamps = np.array([])
    
    # 验证对齐后的数据长度是否一致
    data_lengths = [v.shape[0] for v in aligned_data.values()]
    if len(set(data_lengths)) > 1:
        raise ValueError("对齐后的各模态数据长度不一致")
    
    logger.info(f"Aligned data lengths: {data_lengths}")
    
    return aligned_data, aligned_timestamps




def simple_align(data_dict, timestamps_dict, gap_threshold=1.0):
    # 根据值的长度找到最大长度
    max_length = max(v.shape[0] for v in data_dict.values())
    # 对每个数据进行填充
    for k in data_dict:
        cur_length = data_dict[k].shape[0]
        if cur_length < max_length:
            # 为每个维度指定填充
            pad_width = [(0, max_length-cur_length)]  # 第一维的填充
            for _ in range(len(data_dict[k].shape) - 1):  # 其他维度不填充
                pad_width.append((0, 0))
            
            data_dict[k] = np.pad(data_dict[k], 
                                 pad_width=pad_width,
                                 mode='edge')
    # 采用最长的数据的时间戳
    for k, v in timestamps_dict.items():
        if v.shape[0] == max_length:
            data_dict["timestamp"] = v
            break
    
    return data_dict
