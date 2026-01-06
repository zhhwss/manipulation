import json
import copy
import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
import torch
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import pickle
import random
import gzip
import lz4.frame
import h5py
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# 新增：数据增强函数
def apply_augmentation(frame, aug_type="original"):
    """应用数据增强到frame_color"""
    if aug_type == "original":
        return frame
    
    frame_aug = frame.copy()
    
    if aug_type == "brightness":
        # 亮度调整：随机调整 ±30%
        factor = np.random.uniform(0.7, 1.3)
        frame_aug = np.clip(frame_aug * factor, 0, 255).astype(np.uint8)
    
    elif aug_type == "contrast":
        # 对比度调整
        factor = np.random.uniform(0.8, 1.2)
        mean = frame_aug.mean()
        frame_aug = np.clip((frame_aug - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    elif aug_type == "blur":
        # 随机模糊
        kernel_size = np.random.choice([3, 5, 7])
        frame_aug = cv2.GaussianBlur(frame_aug, (kernel_size, kernel_size), 0)
    
    elif aug_type == "noise":
        # 添加高斯噪声
        noise = np.random.normal(0, np.random.uniform(5, 15), frame_aug.shape)
        frame_aug = np.clip(frame_aug + noise, 0, 255).astype(np.uint8)
    
    elif aug_type == "saturation":
        # 饱和度调整
        hsv = cv2.cvtColor(frame_aug, cv2.COLOR_RGB2HSV).astype(np.float32)
        factor = np.random.uniform(0.8, 1.2)
        hsv[:, :, 1] *= factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        frame_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    elif aug_type == "gamma":
        # Gamma校正
        gamma = np.random.uniform(0.8, 1.2)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        frame_aug = cv2.LUT(frame_aug, table)
    
    return frame_aug

def quantize_frame_gpu(frame, kmeans_centers, device='cuda:0'):
    """GPU加速的量化"""
    h, w = frame.shape[:2]
    frame_tensor = torch.from_numpy(frame).float().to(device).view(-1, 3)
    centers_tensor = torch.from_numpy(kmeans_centers).float().to(device)
    
    chunk_size = 100000
    quantized = torch.zeros_like(frame_tensor)
    for i in range(0, len(frame_tensor), chunk_size):
        chunk = frame_tensor[i:i+chunk_size]
        distances = torch.cdist(chunk, centers_tensor, p=2)
        labels = torch.argmin(distances, dim=1)
        quantized[i:i+chunk_size] = centers_tensor[labels]
    
    return quantized.view(h, w, 3).byte().cpu().numpy()

class RobotGraspDataset(Dataset):
    def __init__(self, data_dir: str, image_type: str = 'depth_mask', third_view: bool = False, image_scale: float = 1.0, 
                 window_size: int = 10, output_type: str = 'ee_pose', action_chunk: int = 50, action_stride: int = 5, data_aug: int = 5, cache=None, color_view: bool = False, 
                 cache_compression: str = 'hdf5'):
        """
        Args:
            data_dir: Directory containing json and mp4 files
            image_type: 'depth_mask' or 'color'
            window_size: Number of frames to use for prediction
            cache_compression: 缓存压缩方式 ('lz4', 'gzip', 'hdf5', 'numpy_chunks', 'none')
        """
        self.data_dir = data_dir
        self.image_type = image_type
        self.third_view = third_view
        self.color_view = color_view
        self.image_scale = image_scale
        self.window_size = window_size
        self.output_type = output_type
        self.action_chunk = action_chunk
        self.action_stride = action_stride
        self.data_aug = data_aug
        self.cache_compression = cache_compression
        
        # 定义增强类型
        # self.augmentation_types = ["original", "brightness", "contrast", "blur", "noise", "saturation", "gamma"]
        self.augmentation_types = ["original", "brightness", "contrast", "saturation"]

        
        # Get all json files
        self.json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        self.json_files.sort()
        with open("/home/zhouhao/work/manipulation/process/kmeans_centers/kmeans_20_centers.pkl", "rb") as f:
            self.kmeans_centers = pickle.load(f).astype(np.float32)   
        # Prepare data
        self.cache = cache
        self.sequences = []
        self._cached = False
        if self.cache is None or not os.path.exists(self.cache):
            self._prepare_sequences()
        else:
            # 尝试使用新的高效加载方法
            if not self._load_sequences_cache_optimized(self.cache):
                # 回退到原始方法
                with open(self.cache, "rb") as f:
                    self.sequences = pickle.load(f)
            self._cached = True

    def _save_cache(self):
        if self.cache is not None and not self._cached:
            print("saving cache to ", self.cache)
            # 使用高效缓存方法
            self.save_sequences_cache_optimized(self.cache, compression=self.cache_compression)

    
    def _prepare_sequences(self):
        """Prepare sequences for training"""
        for eps_index, json_file in enumerate(self.json_files[:4]):
            json_path = os.path.join(self.data_dir, json_file)
            mp4_name = json_file.replace('.json', f'_{self.image_type}.mp4')
            mp4_path = os.path.join(self.data_dir, mp4_name)
            print(mp4_name)
            
            if not os.path.exists(mp4_path):
                print(f"Warning: {mp4_path} not found, skipping {json_file}")
                continue
            

            
            # 为每种增强类型生成数据
            if self.color_view:
                aug_times = 3
            else:
                aug_times = 1
            for _ in range(aug_times):
                with open(json_path, 'r') as f:
                    data = json.load(f)
            # Load video
            cap = cv2.VideoCapture(mp4_path)
            if self.third_view:
                mp4_path_3 = mp4_path.replace('.mp4', '3.mp4')
                cap_3 = cv2.VideoCapture(mp4_path_3)
            else:
                cap_3 = None
            if self.color_view:
                mp4_path_color = json_file.replace('.json', f'_color.mp4')
                mp4_path_color = os.path.join(self.data_dir, mp4_path_color)
                cap_color = cv2.VideoCapture(mp4_path_color)
            else:
                cap_color = None
                mp4_path_color = None

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale if depth_mask
                if self.image_type == 'depth_mask' or self.image_type == 'depth_mask3':
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif self.image_type == 'color' or self.image_type == 'color3':  # color
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (int(frame.shape[1] * self.image_scale), int(frame.shape[0] * self.image_scale)))
                
                    # frame = frame.astype(np.float32) / 255.0  # Scale to 0-1
                # Add channel dimensions
                if len(frame.shape) == 2:
                    frame = frame[np.newaxis, :, :]  # (1, H, W)
                else:
                    frame = np.transpose(frame, (2, 0, 1))  # (3, H, W)
                
                    if cap_3 is not None:
                        ret_3, frame_3 = cap_3.read()
                        if not ret_3:
                            break
                        # Convert to grayscale if depth_mask
                        if self.image_type == 'depth_mask' or self.image_type == 'depth_mask3':
                            if len(frame_3.shape) == 3:
                                frame_3 = cv2.cvtColor(frame_3, cv2.COLOR_BGR2GRAY)
                        elif self.image_type == 'color' or self.image_type == 'color3':  # color
                            frame_3 = cv2.cvtColor(frame_3, cv2.COLOR_BGR2RGB)
                        frame_3 = cv2.resize(frame_3, (int(frame_3.shape[1] * self.image_scale), int(frame_3.shape[0] * self.image_scale)))
                        
                        frame_3 = frame_3.astype(np.float32) / 255.0  # Scale to 0-1
                        # Add channel dimensions
                        if len(frame_3.shape) == 2:
                            frame_3 = frame_3[np.newaxis, :, :]  # (1, H, W)
                        else:
                            frame_3 = np.transpose(frame_3, (2, 0, 1))  # (3, H, W)
                        frame = np.concatenate([frame, frame_3], axis=0)
                    
                    if mp4_path_color is not None:
                        ret_color, frame_color = cap_color.read()
                        if not ret_color:
                            break
                        frame_color = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB)
                        frame_color = cv2.resize(frame_color, (int(frame_color.shape[1] * self.image_scale), int(frame_color.shape[0] * self.image_scale)))
                        
                        # 应用数据增强
                        aug_type = np.random.choice(self.augmentation_types)
                        frame_color = apply_augmentation(frame_color, aug_type)
                        frame_color_gray = cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY)
                        frame_rgb = cv2.adaptiveThreshold(frame_color_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        frame_color = frame_rgb[np.newaxis, :, :]

                        frame = np.concatenate([frame, frame_color], axis=0)
                frames.append(frame)
            
            cap.release()
            if cap_color is not None:
                cap_color.release()
            if cap_3 is not None:
                cap_3.release()
            if len(frames) != len(data):
                print(f"Warning: {json_file} has {len(frames)} frames but {len(data)} data points")
                continue
            
            temp_cnt = 0
            closed_sign = 0
            last_gripper_close = 0
            close_xyz  = None
            closed_index = None
            for i in range(len(data)):
                if last_gripper_close == 1 and data[i]["qpos"][-1] > 0.001:
                    temp_cnt += 1
                if temp_cnt > 30:
                    closed_sign = 1
                if closed_sign == 1 and closed_index is None:
                    closed_index = i

                data[i]["qpos"].append(closed_sign)
                last_gripper_close = data[i]["ee_pose"][-1]
                
                if close_xyz is None and data[i]["ee_pose"][-1] == 1:
                    close_position = np.array(data[i]["position"][:6])
                    close_xyz = forward_kinematics_piper(close_position)
                    # TODO
                    close_xyz[1] += 0.3
                    
            for i in range(len(data)):
                if data[i]["qpos"][-1] == 1: break
                temp = frames[i][:3]
                temp = temp.astype(np.float32) / 255.0
                mask1 = (temp[0] >= close_xyz[0] - 0.15) & (temp[0] <= close_xyz[0] + 0.15) 
                mask2 = (temp[1] >= close_xyz[1] - 0.1) & (temp[1] <= close_xyz[1] + 0.1)
                mask3 = np.zeros_like(temp[0], dtype=bool)
                mask3[160:360, 140:540] = True
                mask = (mask1 & mask2) | mask3
                if closed_index is None or i < closed_index:
                    frames[i][:, ~mask] = 0
                else:
                    # frames[i][:3, ~mask] = 0 # v1
                    mask3[:, 110:570] = True
                    frames[i][:, ~mask3] = 0 # v2

                # temp = frames[i][:3]
                # temp = (temp * 255).astype(np.uint8)
                # temp = np.transpose(temp, (1, 2, 0))
                # cv2.imwrite(f'figs/{mp4_name}_{i}.png', temp)
                stops = []
                stopping = False
                for i in range(len(data)-1):
                    if (np.array(data[i]["ee_pose"]) == np.array(data[i+1]["ee_pose"])).all():
                        stopping = True
                    else:
                        if stopping:
                            stops.append(i)
                        stopping = False
                stops = stops[-2:]
                stops.append(len(data)-1)
                stop_idx = 0
            
            # Create sequences
            for i in range(len(data)):
                # Get window of frames and qpos
                # eps_index = i
                start_idx = max(0, i - self.window_size + 1)
                end_idx = i + 1

                # Target
                targets = []
                for j in range(i, i + self.action_chunk):
                    target = data[min(j, stops[stop_idx])][self.output_type]
                    targets.append(target)
                target = np.concatenate(targets)
                if i == stops[stop_idx]:
                    stop_idx += 1
                
                # Pad with first frame if needed
                if end_idx - start_idx < self.window_size:
                    pad_size = self.window_size - (end_idx - start_idx)
                    frame_window = [frames[start_idx]] * pad_size + frames[start_idx:end_idx]
                    qpos_window = [data[start_idx]['qpos']] * pad_size + [data[j]['qpos'] for j in range(start_idx, end_idx)]
                    self.sequences.append({
                        'index': eps_index,
                        'frames': frame_window,
                        'qpos': qpos_window,
                        'target': target
                    })
                else:
                    frame_window = frames[start_idx:end_idx]
                    qpos_window = [data[j]['qpos'] for j in range(start_idx, end_idx)]
                    self.sequences.append({
                        'index': eps_index,
                            'frames': frame_window,
                            'qpos': qpos_window,
                            'target': target
                        })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Convert to tensors
        index = torch.tensor(seq['index'], dtype=torch.int32)
        frames = torch.tensor(np.array(seq['frames']) / 255.0, dtype=torch.float32)
        qpos = torch.tensor(np.array(seq['qpos']), dtype=torch.float32)
        target = torch.tensor(np.array(seq['target']), dtype=torch.float32)
        
        return index,frames, qpos, target
    
    # ============= 高效缓存方法 =============
    
    def save_sequences_cache_optimized(self, cache_path: str, compression: str = 'lz4', chunk_size: int = 1000):
        """
        高效保存sequences到缓存文件
        
        Args:
            cache_path: 缓存文件路径
            compression: 压缩方式 ('lz4', 'gzip', 'hdf5', 'numpy_chunks', 'none')
            chunk_size: 分块大小 (仅对numpy_chunks有效)
        """
        print(f"Saving {len(self.sequences)} sequences to {cache_path} with {compression} compression")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        
        if compression == 'hdf5':
            self._save_hdf5_cache(cache_path)
        elif compression == 'numpy_chunks':
            self._save_numpy_chunks_cache(cache_path, chunk_size)
        else:
            self._save_compressed_cache(cache_path, compression)
        
        # 显示缓存信息
        info = self._get_cache_info(cache_path, compression)
        if info:
            print(f"Cache saved successfully! Size: {info['size_mb']:.2f} MB")
    
    def _load_sequences_cache_optimized(self, cache_path: str) -> bool:
        """
        高效加载序列缓存
        
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(cache_path):
            return False
        
        try:
            print(f"Loading sequences from {cache_path} with {self.cache_compression} compression")
            
            if self.cache_compression == 'hdf5':
                self.sequences = self._load_hdf5_cache(cache_path)
            elif self.cache_compression == 'numpy_chunks':
                self.sequences = self._load_numpy_chunks_cache(cache_path)
            else:
                self.sequences = self._load_compressed_cache(cache_path)
            
            print(f"Successfully loaded {len(self.sequences)} sequences from cache")
            return True
            
        except Exception as e:
            print(f"Failed to load optimized cache: {e}")
            return False
    
    def _save_compressed_cache(self, cache_path: str, compression: str):
        """保存压缩缓存"""
        # 预处理：优化numpy数组存储
        processed_sequences = []
        for seq in self.sequences:
            processed_seq = {}
            for key, value in seq.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    # 将numpy数组列表转换为单个numpy数组
                    processed_seq[key] = np.array(value)
                else:
                    processed_seq[key] = value
            processed_sequences.append(processed_seq)
        
        if compression == 'lz4':
            # LZ4压缩 - 最快的压缩速度
            data = pickle.dumps(processed_sequences, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_data = lz4.frame.compress(data)
            with open(cache_path, 'wb') as f:
                f.write(compressed_data)
        
        elif compression == 'gzip':
            # Gzip压缩 - 更好的压缩率
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(processed_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif compression == 'pickle_protocol5':
            # Pickle协议5 - 对大数组优化
            with open(cache_path, 'wb') as f:
                pickle.dump(processed_sequences, f, protocol=5)
        
        else:  # 'none'
            # 无压缩
            with open(cache_path, 'wb') as f:
                pickle.dump(processed_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_compressed_cache(self, cache_path: str):
        """加载压缩缓存"""
        if self.cache_compression == 'lz4':
            with open(cache_path, 'rb') as f:
                compressed_data = f.read()
            data = lz4.frame.decompress(compressed_data)
            return pickle.loads(data)
        
        elif self.cache_compression == 'gzip':
            with gzip.open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        else:  # 'pickle_protocol5' or 'none'
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    
    def _save_hdf5_cache(self, cache_path: str):
        """使用HDF5格式保存 - 最适合大型数值数据"""
        with h5py.File(cache_path, 'w') as f:
            f.attrs['num_sequences'] = len(self.sequences)
            
            for i, seq in enumerate(self.sequences):
                grp = f.create_group(f'seq_{i}')
                grp.attrs['index'] = seq['index']
                
                # 保存frames
                frames = np.array(seq['frames'])
                grp.create_dataset('frames', data=frames, compression='gzip', compression_opts=6)
                
                # 保存qpos
                qpos = np.array(seq['qpos'])
                grp.create_dataset('qpos', data=qpos, compression='gzip', compression_opts=6)
                
                # 保存target
                target = np.array(seq['target'])
                grp.create_dataset('target', data=target, compression='gzip', compression_opts=6)
    
    def _load_hdf5_cache(self, cache_path: str):
        """从HDF5格式加载"""
        sequences = []
        with h5py.File(cache_path, 'r') as f:
            num_sequences = f.attrs['num_sequences']
            
            for i in range(num_sequences):
                grp = f[f'seq_{i}']
                seq = {
                    'index': grp.attrs['index'],
                    'frames': grp['frames'][:].tolist(),  # 转回列表格式以保持兼容性
                    'qpos': grp['qpos'][:].tolist(),
                    'target': grp['target'][:]
                }
                sequences.append(seq)
        
        return sequences
    
    def _save_numpy_chunks_cache(self, cache_path: str, chunk_size: int):
        """使用numpy格式分块保存 - 支持并行处理"""
        cache_dir = cache_path + '_chunks'
        os.makedirs(cache_dir, exist_ok=True)
        
        # 保存元数据
        metadata = {
            'num_sequences': len(self.sequences),
            'chunk_size': chunk_size,
            'num_chunks': (len(self.sequences) + chunk_size - 1) // chunk_size
        }
        
        with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # 分块保存函数
        def save_chunk(args):
            chunk_idx, chunk_sequences = args
            chunk_data = {
                'indices': [],
                'frames': [],
                'qpos': [],
                'targets': []
            }
            
            for seq in chunk_sequences:
                chunk_data['indices'].append(seq['index'])
                chunk_data['frames'].append(np.array(seq['frames']))
                chunk_data['qpos'].append(np.array(seq['qpos']))
                chunk_data['targets'].append(np.array(seq['target']))
            
            chunk_path = os.path.join(cache_dir, f'chunk_{chunk_idx}.npz')
            np.savez_compressed(chunk_path, **chunk_data)
            return chunk_idx
        
        # 并行保存chunks
        chunks = []
        for i in range(0, len(self.sequences), chunk_size):
            chunk = self.sequences[i:i + chunk_size]
            chunks.append((i // chunk_size, chunk))
        
        num_workers = min(8, mp.cpu_count())
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(save_chunk, chunks))
    
    def _load_numpy_chunks_cache(self, cache_path: str):
        """从numpy格式分块加载"""
        cache_dir = cache_path + '_chunks'
        
        # 加载元数据
        with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        sequences = []
        
        # 按顺序加载chunks
        for chunk_idx in range(metadata['num_chunks']):
            chunk_path = os.path.join(cache_dir, f'chunk_{chunk_idx}.npz')
            
            with np.load(chunk_path) as data:
                indices = data['indices']
                frames = data['frames']
                qpos = data['qpos']
                targets = data['targets']
                
                for i in range(len(indices)):
                    seq = {
                        'index': int(indices[i]),
                        'frames': frames[i].tolist(),  # 转回列表格式
                        'qpos': qpos[i].tolist(),
                        'target': targets[i]
                    }
                    sequences.append(seq)
        
        return sequences
    
    def _get_cache_info(self, cache_path: str, compression: str):
        """获取缓存文件信息"""
        if not os.path.exists(cache_path):
            return None
        
        if compression == 'numpy_chunks':
            cache_dir = cache_path + '_chunks'
            if os.path.exists(cache_dir):
                total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                               for f in os.listdir(cache_dir))
                size_mb = total_size / (1024 * 1024)
            else:
                size_mb = 0
        else:
            size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        
        return {
            'path': cache_path,
            'size_mb': size_mb,
            'compression': compression
        }

def split_dataset_by_trajectory(full_dataset, train_split: float = 0.8):
    """
    Split dataset by trajectory indices to ensure no trajectory appears in both train and val
    
    Args:
        full_dataset: RobotGraspDataset instance
        train_split: fraction of trajectories to use for training
    
    Returns:
        train_indices, val_indices: lists of indices for train and validation
    """
    # Get all unique trajectory indices
    trajectory_indices = set()
    for seq in full_dataset.sequences:
        trajectory_indices.add(seq['index'])
    
    trajectory_indices = sorted(list(trajectory_indices))
    print(f"Found {len(trajectory_indices)} unique trajectories")
    
    # Split trajectories
    np.random.seed(42)  # For reproducible splits
    np.random.shuffle(trajectory_indices)
    
    n_train_trajectories = int(len(trajectory_indices) * train_split)
    train_trajectories = set(trajectory_indices[:n_train_trajectories])
    val_trajectories = set(trajectory_indices[n_train_trajectories:])
    
    print(f"Train trajectories: {len(train_trajectories)}")
    print(f"Val trajectories: {len(val_trajectories)}")
    
    # Get sequence indices for each split
    train_indices = []
    val_indices = []
    
    for i, seq in enumerate(full_dataset.sequences):
        if seq['index'] in train_trajectories:
            train_indices.append(i)
        else:
            val_indices.append(i)
    
    print(f"Train sequences: {len(train_indices)}")
    print(f"Val sequences: {len(val_indices)}")
    
    return train_indices, val_indices

def create_data_loaders(data_dir: str, image_type: str = 'depth_mask', third_view: bool = False, image_scale: float = 1.0, output_type: str = 'ee_pose', 
                        action_chunk: int = 50, action_stride: int = 5,
                        batch_size: int = 8, train_split: float = 0.8, window_size: int = 10, num_workers: int = 4, data_aug: int = 5, cache=None, color_view: bool = False):
    """
    Create train and validation data loaders
    """
    # Create full dataset
    full_dataset = RobotGraspDataset(data_dir, image_type, third_view, image_scale, window_size, output_type, action_chunk, action_stride, data_aug, cache=cache, color_view=color_view)
    # full_dataset._save_cache()
    
    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset

def create_data_loaders_by_trajectory(data_dir: str, image_type: str = 'depth_mask', third_view: bool = False, image_scale: float = 1.0, output_type: str = 'ee_pose', 
                        action_chunk: int = 50, action_stride: int = 5,
                        batch_size: int = 8, train_split: float = 0.8, window_size: int = 10, num_workers: int = 4, data_aug: int = 5, cache=None, color_view: bool = False):
    """
    Create train and validation data loaders split by trajectory
    """
    # Create full dataset
    full_dataset = RobotGraspDataset(data_dir, image_type, third_view, image_scale, window_size, output_type, action_chunk, action_stride, data_aug, cache=cache, color_view=color_view)
    # full_dataset._save_cache()
    # Split by trajectory
    train_indices, val_indices = split_dataset_by_trajectory(full_dataset, train_split)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset



def forward_kinematics_piper(joint_angles, urdf_path="./urdf/piper.urdf"):
    """
    计算Piper机械臂的正运动学
    Args:
        joint_angles: 6个关节角度 [joint1, joint2, joint3, joint4, joint5, joint6]
        urdf_path: URDF文件路径
    Returns:
        hand_tcp的xyz坐标
    """
    # 读取URDF文件
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # 解析关节链：从base_link到hand_tcp
    joint_chain = []
    current_link = "base_link"
    target_link = "hand_tcp"
    
    # 构建从base_link到hand_tcp的关节链
    while current_link != target_link:
        # 找到以current_link为parent的joint
        for joint in root.findall(".//joint"):
            parent_link = joint.find("parent").get("link")
            if parent_link == current_link:
                joint_name = joint.get("name")
                child_link = joint.find("child").get("link")
                
                # 获取关节变换信息
                origin = joint.find("origin")
                if origin is not None:
                    xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
                    rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
                else:
                    xyz = [0, 0, 0]
                    rpy = [0, 0, 0]
                
                # 获取关节轴
                axis_elem = joint.find("axis")
                if axis_elem is not None:
                    axis = [float(x) for x in axis_elem.get("xyz", "0 0 1").split()]
                else:
                    axis = [0, 0, 1]
                
                joint_chain.append({
                    'name': joint_name,
                    'type': joint.get("type"),
                    'xyz': xyz,
                    'rpy': rpy,
                    'axis': axis,
                    'child_link': child_link
                })
                
                current_link = child_link
                break
        else:
            raise ValueError(f"Cannot find joint with parent link: {current_link}")
    
    # 计算变换矩阵
    def rpy_to_matrix(rpy):
        """将roll, pitch, yaw转换为旋转矩阵"""
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
    
    def create_transform_matrix(xyz, rpy, joint_angle, axis, joint_type):
        """创建关节变换矩阵"""
        # 平移矩阵
        T_trans = np.eye(4)
        T_trans[:3, 3] = xyz
        
        # 旋转矩阵（从URDF的rpy）
        R_rpy = rpy_to_matrix(rpy)
        T_rot = np.eye(4)
        T_rot[:3, :3] = R_rpy
        
        # 关节旋转矩阵
        if joint_type == "revolute":
            # 计算关节轴的单位向量
            axis_norm = np.array(axis) / np.linalg.norm(axis)
            # 使用Rodrigues公式计算旋转矩阵
            K = np.array([[0, -axis_norm[2], axis_norm[1]],
                         [axis_norm[2], 0, -axis_norm[0]],
                         [-axis_norm[1], axis_norm[0], 0]])
            R_joint = np.eye(3) + np.sin(joint_angle) * K + (1 - np.cos(joint_angle)) * (K @ K)
        else:
            R_joint = np.eye(3)
        
        T_joint = np.eye(4)
        T_joint[:3, :3] = R_joint
        
        return T_trans @ T_rot @ T_joint
    
    # 计算从base_link到hand_tcp的变换矩阵
    T = np.eye(4)
    joint_idx = 0
    
    for joint_info in joint_chain:
        if joint_info['type'] == 'revolute' and joint_idx < len(joint_angles):
            joint_angle = joint_angles[joint_idx]
            joint_idx += 1
        else:
            joint_angle = 0  # 对于fixed关节或prismatic关节
        
        T_joint = create_transform_matrix(
            joint_info['xyz'], 
            joint_info['rpy'], 
            joint_angle, 
            joint_info['axis'], 
            joint_info['type']
        )
        T = T @ T_joint
    
    # 计算hand_tcp的xyz坐标
    hand_tcp_homogeneous = T @ np.array([0, 0, 0, 1])
    hand_tcp_xyz = hand_tcp_homogeneous[:3]
    
    return hand_tcp_xyz
