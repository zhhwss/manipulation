#!/usr/bin/env python3
"""
RobotGraspDataset 高效缓存使用示例

这个文件展示了如何使用新的高效缓存方法来保存和加载 RobotGraspDataset 的 sequences 数据。
"""

from data_loader import RobotGraspDataset
import os
import time

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 1. 创建数据集（首次处理）
    dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        image_type='depth_mask',
        window_size=10,
        action_chunk=50,
        cache=None,  # 不使用缓存，重新处理数据
        cache_compression='lz4'  # 设置缓存压缩方式
    )
    
    print(f"处理完成，共 {len(dataset.sequences)} 个序列")
    
    # 2. 手动保存缓存
    cache_path = "my_sequences_cache.lz4"
    dataset.save_sequences_cache_optimized(cache_path, compression='lz4')
    
    # 3. 下次使用时，直接从缓存加载
    dataset_from_cache = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        cache=cache_path,
        cache_compression='lz4'
    )
    
    print(f"从缓存加载，共 {len(dataset_from_cache.sequences)} 个序列")

def example_different_compressions():
    """不同压缩方法示例"""
    print("\n=== 不同压缩方法示例 ===")
    
    # 假设已有数据集
    dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        cache=None
    )
    
    # 测试不同压缩方法
    compressions = {
        'lz4': '最快的压缩/解压速度',
        'gzip': '更好的压缩率',
        'hdf5': '最适合大型数值数据',
        'numpy_chunks': '支持并行处理和大数据集'
    }
    
    for compression, description in compressions.items():
        print(f"\n--- {compression.upper()}: {description} ---")
        
        cache_path = f"cache_{compression}"
        
        # 保存
        start_time = time.time()
        dataset.save_sequences_cache_optimized(cache_path, compression=compression)
        save_time = time.time() - start_time
        
        # 获取文件大小
        if compression == 'numpy_chunks':
            cache_dir = cache_path + '_chunks'
            if os.path.exists(cache_dir):
                total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                               for f in os.listdir(cache_dir))
                file_size_mb = total_size / (1024 * 1024)
            else:
                file_size_mb = 0
        else:
            file_size_mb = os.path.getsize(cache_path) / (1024 * 1024) if os.path.exists(cache_path) else 0
        
        print(f"保存时间: {save_time:.2f}s, 文件大小: {file_size_mb:.2f} MB")

def example_automatic_cache():
    """自动缓存示例"""
    print("\n=== 自动缓存示例 ===")
    
    # 使用自动缓存：如果缓存文件存在就加载，否则创建并保存
    cache_file = "auto_cache.lz4"
    
    dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        image_type='depth_mask',
        cache=cache_file,  # 自动使用缓存
        cache_compression='lz4'
    )
    
    # 如果是首次运行，数据会被处理并自动保存到缓存
    # 如果缓存已存在，会直接从缓存加载
    
    print(f"数据集准备就绪，共 {len(dataset.sequences)} 个序列")

def example_production_workflow():
    """生产环境工作流示例"""
    print("\n=== 生产环境工作流示例 ===")
    
    # 步骤1: 数据预处理阶段（一次性）
    print("1. 数据预处理阶段...")
    
    preprocessing_dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        image_type='depth_mask',
        window_size=10,
        action_chunk=50,
        cache=None,  # 强制重新处理
        cache_compression='gzip'  # 生产环境推荐gzip平衡压缩率和速度
    )
    
    # 保存预处理好的缓存
    production_cache = "production_cache.gz"
    preprocessing_dataset.save_sequences_cache_optimized(production_cache, compression='gzip')
    
    print(f"预处理完成，缓存保存到 {production_cache}")
    
    # 步骤2: 训练阶段（多次使用）
    print("\n2. 训练阶段...")
    
    for epoch in range(3):  # 模拟3个epoch
        print(f"Epoch {epoch + 1}: 加载数据...")
        
        start_time = time.time()
        training_dataset = RobotGraspDataset(
            data_dir="../lhx/piper_real_cnn/data/20250808",
            cache=production_cache,
            cache_compression='gzip'
        )
        load_time = time.time() - start_time
        
        print(f"  数据加载完成，用时 {load_time:.2f}s，共 {len(training_dataset.sequences)} 个序列")
        print(f"  开始训练...")
        # 这里进行实际的训练逻辑
        time.sleep(0.5)  # 模拟训练时间
        print(f"  Epoch {epoch + 1} 训练完成")

def example_large_dataset_handling():
    """大数据集处理示例"""
    print("\n=== 大数据集处理示例 ===")
    
    # 对于非常大的数据集，使用numpy_chunks方式
    print("处理大数据集...")
    
    large_dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        cache=None,
        cache_compression='numpy_chunks'
    )
    
    # 使用分块保存，支持并行处理
    large_cache = "large_dataset_cache"
    large_dataset.save_sequences_cache_optimized(
        large_cache, 
        compression='numpy_chunks',
        chunk_size=500  # 每个chunk包含500个序列
    )
    
    print("大数据集缓存保存完成")
    
    # 加载时也是自动处理分块
    loaded_large_dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        cache=large_cache,
        cache_compression='numpy_chunks'
    )
    
    print(f"大数据集加载完成，共 {len(loaded_large_dataset.sequences)} 个序列")

def example_cache_migration():
    """缓存迁移示例"""
    print("\n=== 缓存迁移示例 ===")
    
    # 假设你有一个旧的pickle缓存，想转换为高效格式
    old_cache = "old_sequences.pkl"
    
    # 如果存在旧缓存，转换为新格式
    if os.path.exists(old_cache):
        print("发现旧缓存，开始迁移...")
        
        # 加载旧缓存
        dataset = RobotGraspDataset(
            data_dir="../lhx/piper_real_cnn/data/20250808",
            cache=old_cache,
            cache_compression='none'  # 旧格式
        )
        
        # 保存为新格式
        new_cache = "migrated_cache.lz4"
        dataset.save_sequences_cache_optimized(new_cache, compression='lz4')
        
        print(f"缓存迁移完成: {old_cache} -> {new_cache}")
        
        # 验证新缓存
        migrated_dataset = RobotGraspDataset(
            data_dir="../lhx/piper_real_cnn/data/20250808",
            cache=new_cache,
            cache_compression='lz4'
        )
        
        print(f"验证完成，数据一致性: {len(dataset.sequences) == len(migrated_dataset.sequences)}")

def cleanup_example_files():
    """清理示例文件"""
    print("\n=== 清理示例文件 ===")
    
    files_to_clean = [
        "my_sequences_cache.lz4",
        "cache_lz4",
        "cache_gzip", 
        "cache_hdf5",
        "auto_cache.lz4",
        "production_cache.gz",
        "migrated_cache.lz4"
    ]
    
    dirs_to_clean = [
        "cache_numpy_chunks_chunks",
        "large_dataset_cache_chunks"
    ]
    
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"删除文件: {file_path}")
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
            print(f"删除目录: {dir_path}")
    
    print("清理完成!")

if __name__ == "__main__":
    print("RobotGraspDataset 高效缓存使用示例")
    print("=" * 50)
    
    try:
        # 运行所有示例
        example_basic_usage()
        example_different_compressions()
        example_automatic_cache()
        example_production_workflow()
        example_large_dataset_handling()
        example_cache_migration()
        
    except Exception as e:
        print(f"示例运行时出错: {e}")
        print("请确保数据目录存在或调整路径")
    
    finally:
        # 清理示例文件
        cleanup_example_files()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成!")
    print("\n推荐用法总结:")
    print("- 开发/调试: compression='lz4'")
    print("- 生产环境: compression='gzip'") 
    print("- 大型数据集: compression='hdf5'")
    print("- 超大数据集: compression='numpy_chunks'") 
 
"""
RobotGraspDataset 高效缓存使用示例

这个文件展示了如何使用新的高效缓存方法来保存和加载 RobotGraspDataset 的 sequences 数据。
"""

from data_loader import RobotGraspDataset
import os
import time

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 1. 创建数据集（首次处理）
    dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        image_type='depth_mask',
        window_size=10,
        action_chunk=50,
        cache=None,  # 不使用缓存，重新处理数据
        cache_compression='lz4'  # 设置缓存压缩方式
    )
    
    print(f"处理完成，共 {len(dataset.sequences)} 个序列")
    
    # 2. 手动保存缓存
    cache_path = "my_sequences_cache.lz4"
    dataset.save_sequences_cache_optimized(cache_path, compression='lz4')
    
    # 3. 下次使用时，直接从缓存加载
    dataset_from_cache = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        cache=cache_path,
        cache_compression='lz4'
    )
    
    print(f"从缓存加载，共 {len(dataset_from_cache.sequences)} 个序列")

def example_different_compressions():
    """不同压缩方法示例"""
    print("\n=== 不同压缩方法示例 ===")
    
    # 假设已有数据集
    dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        cache=None
    )
    
    # 测试不同压缩方法
    compressions = {
        'lz4': '最快的压缩/解压速度',
        'gzip': '更好的压缩率',
        'hdf5': '最适合大型数值数据',
        'numpy_chunks': '支持并行处理和大数据集'
    }
    
    for compression, description in compressions.items():
        print(f"\n--- {compression.upper()}: {description} ---")
        
        cache_path = f"cache_{compression}"
        
        # 保存
        start_time = time.time()
        dataset.save_sequences_cache_optimized(cache_path, compression=compression)
        save_time = time.time() - start_time
        
        # 获取文件大小
        if compression == 'numpy_chunks':
            cache_dir = cache_path + '_chunks'
            if os.path.exists(cache_dir):
                total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                               for f in os.listdir(cache_dir))
                file_size_mb = total_size / (1024 * 1024)
            else:
                file_size_mb = 0
        else:
            file_size_mb = os.path.getsize(cache_path) / (1024 * 1024) if os.path.exists(cache_path) else 0
        
        print(f"保存时间: {save_time:.2f}s, 文件大小: {file_size_mb:.2f} MB")

def example_automatic_cache():
    """自动缓存示例"""
    print("\n=== 自动缓存示例 ===")
    
    # 使用自动缓存：如果缓存文件存在就加载，否则创建并保存
    cache_file = "auto_cache.lz4"
    
    dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        image_type='depth_mask',
        cache=cache_file,  # 自动使用缓存
        cache_compression='lz4'
    )
    
    # 如果是首次运行，数据会被处理并自动保存到缓存
    # 如果缓存已存在，会直接从缓存加载
    
    print(f"数据集准备就绪，共 {len(dataset.sequences)} 个序列")

def example_production_workflow():
    """生产环境工作流示例"""
    print("\n=== 生产环境工作流示例 ===")
    
    # 步骤1: 数据预处理阶段（一次性）
    print("1. 数据预处理阶段...")
    
    preprocessing_dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        image_type='depth_mask',
        window_size=10,
        action_chunk=50,
        cache=None,  # 强制重新处理
        cache_compression='gzip'  # 生产环境推荐gzip平衡压缩率和速度
    )
    
    # 保存预处理好的缓存
    production_cache = "production_cache.gz"
    preprocessing_dataset.save_sequences_cache_optimized(production_cache, compression='gzip')
    
    print(f"预处理完成，缓存保存到 {production_cache}")
    
    # 步骤2: 训练阶段（多次使用）
    print("\n2. 训练阶段...")
    
    for epoch in range(3):  # 模拟3个epoch
        print(f"Epoch {epoch + 1}: 加载数据...")
        
        start_time = time.time()
        training_dataset = RobotGraspDataset(
            data_dir="../lhx/piper_real_cnn/data/20250808",
            cache=production_cache,
            cache_compression='gzip'
        )
        load_time = time.time() - start_time
        
        print(f"  数据加载完成，用时 {load_time:.2f}s，共 {len(training_dataset.sequences)} 个序列")
        print(f"  开始训练...")
        # 这里进行实际的训练逻辑
        time.sleep(0.5)  # 模拟训练时间
        print(f"  Epoch {epoch + 1} 训练完成")

def example_large_dataset_handling():
    """大数据集处理示例"""
    print("\n=== 大数据集处理示例 ===")
    
    # 对于非常大的数据集，使用numpy_chunks方式
    print("处理大数据集...")
    
    large_dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        cache=None,
        cache_compression='numpy_chunks'
    )
    
    # 使用分块保存，支持并行处理
    large_cache = "large_dataset_cache"
    large_dataset.save_sequences_cache_optimized(
        large_cache, 
        compression='numpy_chunks',
        chunk_size=500  # 每个chunk包含500个序列
    )
    
    print("大数据集缓存保存完成")
    
    # 加载时也是自动处理分块
    loaded_large_dataset = RobotGraspDataset(
        data_dir="../lhx/piper_real_cnn/data/20250808",
        cache=large_cache,
        cache_compression='numpy_chunks'
    )
    
    print(f"大数据集加载完成，共 {len(loaded_large_dataset.sequences)} 个序列")

def example_cache_migration():
    """缓存迁移示例"""
    print("\n=== 缓存迁移示例 ===")
    
    # 假设你有一个旧的pickle缓存，想转换为高效格式
    old_cache = "old_sequences.pkl"
    
    # 如果存在旧缓存，转换为新格式
    if os.path.exists(old_cache):
        print("发现旧缓存，开始迁移...")
        
        # 加载旧缓存
        dataset = RobotGraspDataset(
            data_dir="../lhx/piper_real_cnn/data/20250808",
            cache=old_cache,
            cache_compression='none'  # 旧格式
        )
        
        # 保存为新格式
        new_cache = "migrated_cache.lz4"
        dataset.save_sequences_cache_optimized(new_cache, compression='lz4')
        
        print(f"缓存迁移完成: {old_cache} -> {new_cache}")
        
        # 验证新缓存
        migrated_dataset = RobotGraspDataset(
            data_dir="../lhx/piper_real_cnn/data/20250808",
            cache=new_cache,
            cache_compression='lz4'
        )
        
        print(f"验证完成，数据一致性: {len(dataset.sequences) == len(migrated_dataset.sequences)}")

def cleanup_example_files():
    """清理示例文件"""
    print("\n=== 清理示例文件 ===")
    
    files_to_clean = [
        "my_sequences_cache.lz4",
        "cache_lz4",
        "cache_gzip", 
        "cache_hdf5",
        "auto_cache.lz4",
        "production_cache.gz",
        "migrated_cache.lz4"
    ]
    
    dirs_to_clean = [
        "cache_numpy_chunks_chunks",
        "large_dataset_cache_chunks"
    ]
    
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"删除文件: {file_path}")
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
            print(f"删除目录: {dir_path}")
    
    print("清理完成!")

if __name__ == "__main__":
    print("RobotGraspDataset 高效缓存使用示例")
    print("=" * 50)
    
    try:
        # 运行所有示例
        example_basic_usage()
        example_different_compressions()
        example_automatic_cache()
        example_production_workflow()
        example_large_dataset_handling()
        example_cache_migration()
        
    except Exception as e:
        print(f"示例运行时出错: {e}")
        print("请确保数据目录存在或调整路径")
    
    finally:
        # 清理示例文件
        cleanup_example_files()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成!")
    print("\n推荐用法总结:")
    print("- 开发/调试: compression='lz4'")
    print("- 生产环境: compression='gzip'") 
    print("- 大型数据集: compression='hdf5'")
    print("- 超大数据集: compression='numpy_chunks'") 
 