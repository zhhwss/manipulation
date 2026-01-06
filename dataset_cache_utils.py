import pickle
import gzip
import lz4.frame
import numpy as np
import torch
import os
from typing import List, Dict, Any
import h5py
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

class SequenceCacheManager:
    """高效的序列缓存管理器"""
    
    def __init__(self, compression='lz4', num_workers=None):
        """
        Args:
            compression: 压缩方式 ('none', 'gzip', 'lz4', 'pickle_protocol5')
            num_workers: 并行处理的工作进程数
        """
        self.compression = compression
        self.num_workers = num_workers or min(8, mp.cpu_count())
    
    def save_sequences_optimized(self, sequences: List[Dict], cache_path: str, 
                                chunk_size: int = 1000):
        """
        高效保存序列数据
        
        Args:
            sequences: 序列数据列表
            cache_path: 缓存文件路径
            chunk_size: 分块大小，用于并行处理
        """
        print(f"Saving {len(sequences)} sequences to {cache_path}")
        print(f"Using compression: {self.compression}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        
        if self.compression == 'hdf5':
            self._save_hdf5(sequences, cache_path)
        elif self.compression == 'numpy_chunks':
            self._save_numpy_chunks(sequences, cache_path, chunk_size)
        else:
            self._save_compressed(sequences, cache_path)
    
    def load_sequences_optimized(self, cache_path: str) -> List[Dict]:
        """
        高效加载序列数据
        
        Args:
            cache_path: 缓存文件路径
            
        Returns:
            序列数据列表
        """
        if not os.path.exists(cache_path):
            return None
        
        print(f"Loading sequences from {cache_path}")
        
        if self.compression == 'hdf5':
            return self._load_hdf5(cache_path)
        elif self.compression == 'numpy_chunks':
            return self._load_numpy_chunks(cache_path)
        else:
            return self._load_compressed(cache_path)
    
    def _save_compressed(self, sequences: List[Dict], cache_path: str):
        """使用压缩算法保存"""
        # 预处理：将numpy数组转换为列表以减少pickle开销
        processed_sequences = []
        for seq in sequences:
            processed_seq = {}
            for key, value in seq.items():
                if isinstance(value, list) and len(value) > 0:
                    # 检查是否为numpy数组列表
                    if isinstance(value[0], np.ndarray):
                        # 将numpy数组列表转换为单个numpy数组
                        processed_seq[key] = np.array(value)
                    else:
                        processed_seq[key] = value
                else:
                    processed_seq[key] = value
            processed_sequences.append(processed_seq)
        
        if self.compression == 'lz4':
            # LZ4压缩 - 最快的压缩速度
            data = pickle.dumps(processed_sequences, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_data = lz4.frame.compress(data)
            with open(cache_path, 'wb') as f:
                f.write(compressed_data)
        
        elif self.compression == 'gzip':
            # Gzip压缩 - 更好的压缩率
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(processed_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif self.compression == 'pickle_protocol5':
            # Pickle协议5 - 对大数组优化
            with open(cache_path, 'wb') as f:
                pickle.dump(processed_sequences, f, protocol=5)
        
        else:  # 'none'
            # 无压缩
            with open(cache_path, 'wb') as f:
                pickle.dump(processed_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_compressed(self, cache_path: str) -> List[Dict]:
        """加载压缩数据"""
        if self.compression == 'lz4':
            with open(cache_path, 'rb') as f:
                compressed_data = f.read()
            data = lz4.frame.decompress(compressed_data)
            return pickle.loads(data)
        
        elif self.compression == 'gzip':
            with gzip.open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        else:  # 'pickle_protocol5' or 'none'
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    
    def _save_hdf5(self, sequences: List[Dict], cache_path: str):
        """使用HDF5格式保存 - 对大型数值数据最优"""
        with h5py.File(cache_path, 'w') as f:
            f.attrs['num_sequences'] = len(sequences)
            
            for i, seq in enumerate(sequences):
                grp = f.create_group(f'seq_{i}')
                grp.attrs['index'] = seq['index']
                
                # 保存frames (numpy数组)
                frames = np.array(seq['frames'])
                grp.create_dataset('frames', data=frames, compression='gzip', compression_opts=6)
                
                # 保存qpos
                qpos = np.array(seq['qpos'])
                grp.create_dataset('qpos', data=qpos, compression='gzip', compression_opts=6)
                
                # 保存target
                target = np.array(seq['target'])
                grp.create_dataset('target', data=target, compression='gzip', compression_opts=6)
    
    def _load_hdf5(self, cache_path: str) -> List[Dict]:
        """从HDF5格式加载"""
        sequences = []
        with h5py.File(cache_path, 'r') as f:
            num_sequences = f.attrs['num_sequences']
            
            for i in range(num_sequences):
                grp = f[f'seq_{i}']
                seq = {
                    'index': grp.attrs['index'],
                    'frames': grp['frames'][:],
                    'qpos': grp['qpos'][:],
                    'target': grp['target'][:]
                }
                sequences.append(seq)
        
        return sequences
    
    def _save_numpy_chunks(self, sequences: List[Dict], cache_path: str, chunk_size: int):
        """使用numpy格式分块保存"""
        cache_dir = cache_path + '_chunks'
        os.makedirs(cache_dir, exist_ok=True)
        
        # 保存元数据
        metadata = {
            'num_sequences': len(sequences),
            'chunk_size': chunk_size,
            'num_chunks': (len(sequences) + chunk_size - 1) // chunk_size
        }
        
        with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # 分块保存
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
        for i in range(0, len(sequences), chunk_size):
            chunk = sequences[i:i + chunk_size]
            chunks.append((i // chunk_size, chunk))
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            list(executor.map(save_chunk, chunks))
    
    def _load_numpy_chunks(self, cache_path: str) -> List[Dict]:
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
    
    def get_cache_info(self, cache_path: str) -> Dict[str, Any]:
        """获取缓存文件信息"""
        if not os.path.exists(cache_path):
            return None
        
        info = {
            'path': cache_path,
            'size_mb': os.path.getsize(cache_path) / (1024 * 1024),
            'compression': self.compression
        }
        
        if self.compression == 'numpy_chunks':
            cache_dir = cache_path + '_chunks'
            if os.path.exists(cache_dir):
                total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                               for f in os.listdir(cache_dir))
                info['size_mb'] = total_size / (1024 * 1024)
        
        return info

# 为RobotGraspDataset添加缓存方法
def add_cache_methods_to_dataset():
    """为RobotGraspDataset类添加缓存方法"""
    
    def save_sequences_cache(self, cache_path: str, compression='lz4', chunk_size=1000):
        """
        保存sequences到缓存文件
        
        Args:
            cache_path: 缓存文件路径
            compression: 压缩方式 ('lz4', 'gzip', 'hdf5', 'numpy_chunks', 'none')
            chunk_size: 分块大小
        """
        cache_manager = SequenceCacheManager(compression=compression)
        cache_manager.save_sequences_optimized(self.sequences, cache_path, chunk_size)
        print(f"Sequences cache saved to: {cache_path}")
        
        # 显示缓存信息
        info = cache_manager.get_cache_info(cache_path)
        if info:
            print(f"Cache size: {info['size_mb']:.2f} MB")
    
    def load_sequences_cache(self, cache_path: str, compression='lz4'):
        """
        从缓存文件加载sequences
        
        Args:
            cache_path: 缓存文件路径
            compression: 压缩方式
            
        Returns:
            bool: 是否成功加载
        """
        cache_manager = SequenceCacheManager(compression=compression)
        sequences = cache_manager.load_sequences_optimized(cache_path)
        
        if sequences is not None:
            self.sequences = sequences
            print(f"Loaded {len(sequences)} sequences from cache: {cache_path}")
            return True
        else:
            print(f"Cache file not found: {cache_path}")
            return False
    
    # 动态添加方法到类
    return save_sequences_cache, load_sequences_cache

# 使用示例函数
def demo_cache_usage():
    """演示缓存使用方法"""
    print("=== 缓存使用演示 ===")
    
    # 假设已有dataset实例
    # dataset = RobotGraspDataset(...)
    
    # 方法1: 直接使用缓存管理器
    cache_manager = SequenceCacheManager(compression='lz4')
    
    # 保存 (假设sequences已存在)
    # cache_manager.save_sequences_optimized(dataset.sequences, 'cache/sequences_lz4.cache')
    
    # 加载
    # sequences = cache_manager.load_sequences_optimized('cache/sequences_lz4.cache')
    
    # 方法2: 为数据集类添加方法
    save_method, load_method = add_cache_methods_to_dataset()
    
    # 将方法绑定到类 (在实际使用时)
    # RobotGraspDataset.save_sequences_cache = save_method
    # RobotGraspDataset.load_sequences_cache = load_method
    
    print("缓存方法已准备就绪!")

if __name__ == "__main__":
    demo_cache_usage() 
 
import gzip
import lz4.frame
import numpy as np
import torch
import os
from typing import List, Dict, Any
import h5py
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

class SequenceCacheManager:
    """高效的序列缓存管理器"""
    
    def __init__(self, compression='lz4', num_workers=None):
        """
        Args:
            compression: 压缩方式 ('none', 'gzip', 'lz4', 'pickle_protocol5')
            num_workers: 并行处理的工作进程数
        """
        self.compression = compression
        self.num_workers = num_workers or min(8, mp.cpu_count())
    
    def save_sequences_optimized(self, sequences: List[Dict], cache_path: str, 
                                chunk_size: int = 1000):
        """
        高效保存序列数据
        
        Args:
            sequences: 序列数据列表
            cache_path: 缓存文件路径
            chunk_size: 分块大小，用于并行处理
        """
        print(f"Saving {len(sequences)} sequences to {cache_path}")
        print(f"Using compression: {self.compression}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        
        if self.compression == 'hdf5':
            self._save_hdf5(sequences, cache_path)
        elif self.compression == 'numpy_chunks':
            self._save_numpy_chunks(sequences, cache_path, chunk_size)
        else:
            self._save_compressed(sequences, cache_path)
    
    def load_sequences_optimized(self, cache_path: str) -> List[Dict]:
        """
        高效加载序列数据
        
        Args:
            cache_path: 缓存文件路径
            
        Returns:
            序列数据列表
        """
        if not os.path.exists(cache_path):
            return None
        
        print(f"Loading sequences from {cache_path}")
        
        if self.compression == 'hdf5':
            return self._load_hdf5(cache_path)
        elif self.compression == 'numpy_chunks':
            return self._load_numpy_chunks(cache_path)
        else:
            return self._load_compressed(cache_path)
    
    def _save_compressed(self, sequences: List[Dict], cache_path: str):
        """使用压缩算法保存"""
        # 预处理：将numpy数组转换为列表以减少pickle开销
        processed_sequences = []
        for seq in sequences:
            processed_seq = {}
            for key, value in seq.items():
                if isinstance(value, list) and len(value) > 0:
                    # 检查是否为numpy数组列表
                    if isinstance(value[0], np.ndarray):
                        # 将numpy数组列表转换为单个numpy数组
                        processed_seq[key] = np.array(value)
                    else:
                        processed_seq[key] = value
                else:
                    processed_seq[key] = value
            processed_sequences.append(processed_seq)
        
        if self.compression == 'lz4':
            # LZ4压缩 - 最快的压缩速度
            data = pickle.dumps(processed_sequences, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_data = lz4.frame.compress(data)
            with open(cache_path, 'wb') as f:
                f.write(compressed_data)
        
        elif self.compression == 'gzip':
            # Gzip压缩 - 更好的压缩率
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(processed_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif self.compression == 'pickle_protocol5':
            # Pickle协议5 - 对大数组优化
            with open(cache_path, 'wb') as f:
                pickle.dump(processed_sequences, f, protocol=5)
        
        else:  # 'none'
            # 无压缩
            with open(cache_path, 'wb') as f:
                pickle.dump(processed_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_compressed(self, cache_path: str) -> List[Dict]:
        """加载压缩数据"""
        if self.compression == 'lz4':
            with open(cache_path, 'rb') as f:
                compressed_data = f.read()
            data = lz4.frame.decompress(compressed_data)
            return pickle.loads(data)
        
        elif self.compression == 'gzip':
            with gzip.open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        else:  # 'pickle_protocol5' or 'none'
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    
    def _save_hdf5(self, sequences: List[Dict], cache_path: str):
        """使用HDF5格式保存 - 对大型数值数据最优"""
        with h5py.File(cache_path, 'w') as f:
            f.attrs['num_sequences'] = len(sequences)
            
            for i, seq in enumerate(sequences):
                grp = f.create_group(f'seq_{i}')
                grp.attrs['index'] = seq['index']
                
                # 保存frames (numpy数组)
                frames = np.array(seq['frames'])
                grp.create_dataset('frames', data=frames, compression='gzip', compression_opts=6)
                
                # 保存qpos
                qpos = np.array(seq['qpos'])
                grp.create_dataset('qpos', data=qpos, compression='gzip', compression_opts=6)
                
                # 保存target
                target = np.array(seq['target'])
                grp.create_dataset('target', data=target, compression='gzip', compression_opts=6)
    
    def _load_hdf5(self, cache_path: str) -> List[Dict]:
        """从HDF5格式加载"""
        sequences = []
        with h5py.File(cache_path, 'r') as f:
            num_sequences = f.attrs['num_sequences']
            
            for i in range(num_sequences):
                grp = f[f'seq_{i}']
                seq = {
                    'index': grp.attrs['index'],
                    'frames': grp['frames'][:],
                    'qpos': grp['qpos'][:],
                    'target': grp['target'][:]
                }
                sequences.append(seq)
        
        return sequences
    
    def _save_numpy_chunks(self, sequences: List[Dict], cache_path: str, chunk_size: int):
        """使用numpy格式分块保存"""
        cache_dir = cache_path + '_chunks'
        os.makedirs(cache_dir, exist_ok=True)
        
        # 保存元数据
        metadata = {
            'num_sequences': len(sequences),
            'chunk_size': chunk_size,
            'num_chunks': (len(sequences) + chunk_size - 1) // chunk_size
        }
        
        with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # 分块保存
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
        for i in range(0, len(sequences), chunk_size):
            chunk = sequences[i:i + chunk_size]
            chunks.append((i // chunk_size, chunk))
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            list(executor.map(save_chunk, chunks))
    
    def _load_numpy_chunks(self, cache_path: str) -> List[Dict]:
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
    
    def get_cache_info(self, cache_path: str) -> Dict[str, Any]:
        """获取缓存文件信息"""
        if not os.path.exists(cache_path):
            return None
        
        info = {
            'path': cache_path,
            'size_mb': os.path.getsize(cache_path) / (1024 * 1024),
            'compression': self.compression
        }
        
        if self.compression == 'numpy_chunks':
            cache_dir = cache_path + '_chunks'
            if os.path.exists(cache_dir):
                total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                               for f in os.listdir(cache_dir))
                info['size_mb'] = total_size / (1024 * 1024)
        
        return info

# 为RobotGraspDataset添加缓存方法
def add_cache_methods_to_dataset():
    """为RobotGraspDataset类添加缓存方法"""
    
    def save_sequences_cache(self, cache_path: str, compression='lz4', chunk_size=1000):
        """
        保存sequences到缓存文件
        
        Args:
            cache_path: 缓存文件路径
            compression: 压缩方式 ('lz4', 'gzip', 'hdf5', 'numpy_chunks', 'none')
            chunk_size: 分块大小
        """
        cache_manager = SequenceCacheManager(compression=compression)
        cache_manager.save_sequences_optimized(self.sequences, cache_path, chunk_size)
        print(f"Sequences cache saved to: {cache_path}")
        
        # 显示缓存信息
        info = cache_manager.get_cache_info(cache_path)
        if info:
            print(f"Cache size: {info['size_mb']:.2f} MB")
    
    def load_sequences_cache(self, cache_path: str, compression='lz4'):
        """
        从缓存文件加载sequences
        
        Args:
            cache_path: 缓存文件路径
            compression: 压缩方式
            
        Returns:
            bool: 是否成功加载
        """
        cache_manager = SequenceCacheManager(compression=compression)
        sequences = cache_manager.load_sequences_optimized(cache_path)
        
        if sequences is not None:
            self.sequences = sequences
            print(f"Loaded {len(sequences)} sequences from cache: {cache_path}")
            return True
        else:
            print(f"Cache file not found: {cache_path}")
            return False
    
    # 动态添加方法到类
    return save_sequences_cache, load_sequences_cache

# 使用示例函数
def demo_cache_usage():
    """演示缓存使用方法"""
    print("=== 缓存使用演示 ===")
    
    # 假设已有dataset实例
    # dataset = RobotGraspDataset(...)
    
    # 方法1: 直接使用缓存管理器
    cache_manager = SequenceCacheManager(compression='lz4')
    
    # 保存 (假设sequences已存在)
    # cache_manager.save_sequences_optimized(dataset.sequences, 'cache/sequences_lz4.cache')
    
    # 加载
    # sequences = cache_manager.load_sequences_optimized('cache/sequences_lz4.cache')
    
    # 方法2: 为数据集类添加方法
    save_method, load_method = add_cache_methods_to_dataset()
    
    # 将方法绑定到类 (在实际使用时)
    # RobotGraspDataset.save_sequences_cache = save_method
    # RobotGraspDataset.load_sequences_cache = load_method
    
    print("缓存方法已准备就绪!")

if __name__ == "__main__":
    demo_cache_usage() 
 