#!/usr/bin/env python3
"""
安装缓存功能所需的依赖包
"""

import subprocess
import sys
import importlib

def check_and_install_package(package_name, import_name=None, pip_name=None):
    """检查并安装包"""
    if import_name is None:
        import_name = package_name
    if pip_name is None:
        pip_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name} 已安装")
        return True
    except ImportError:
        print(f"✗ {package_name} 未安装，正在安装...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✓ {package_name} 安装成功")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ {package_name} 安装失败")
            return False

def main():
    """主函数"""
    print("检查并安装缓存功能依赖包...")
    print("=" * 40)
    
    packages = [
        ("lz4", "lz4.frame", "lz4"),
        ("h5py", "h5py", "h5py"),
        ("numpy", "numpy", "numpy"),
        ("opencv-python", "cv2", "opencv-python")
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package_name, import_name, pip_name in packages:
        if check_and_install_package(package_name, import_name, pip_name):
            success_count += 1
    
    print("=" * 40)
    print(f"安装结果: {success_count}/{total_count} 个包安装成功")
    
    if success_count == total_count:
        print("✓ 所有依赖包已准备就绪!")
        print("\n现在您可以使用高效缓存功能:")
        print("- LZ4 压缩 (最快)")
        print("- HDF5 格式 (最适合数值数据)")
        print("- 并行分块处理")
    else:
        print("✗ 部分依赖包安装失败，请手动安装")
        
    print("\n使用示例:")
    print("python cache_usage_examples.py")
    print("python test_cache_performance.py")

if __name__ == "__main__":
    main() 
 
"""
安装缓存功能所需的依赖包
"""

import subprocess
import sys
import importlib

def check_and_install_package(package_name, import_name=None, pip_name=None):
    """检查并安装包"""
    if import_name is None:
        import_name = package_name
    if pip_name is None:
        pip_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name} 已安装")
        return True
    except ImportError:
        print(f"✗ {package_name} 未安装，正在安装...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✓ {package_name} 安装成功")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ {package_name} 安装失败")
            return False

def main():
    """主函数"""
    print("检查并安装缓存功能依赖包...")
    print("=" * 40)
    
    packages = [
        ("lz4", "lz4.frame", "lz4"),
        ("h5py", "h5py", "h5py"),
        ("numpy", "numpy", "numpy"),
        ("opencv-python", "cv2", "opencv-python")
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package_name, import_name, pip_name in packages:
        if check_and_install_package(package_name, import_name, pip_name):
            success_count += 1
    
    print("=" * 40)
    print(f"安装结果: {success_count}/{total_count} 个包安装成功")
    
    if success_count == total_count:
        print("✓ 所有依赖包已准备就绪!")
        print("\n现在您可以使用高效缓存功能:")
        print("- LZ4 压缩 (最快)")
        print("- HDF5 格式 (最适合数值数据)")
        print("- 并行分块处理")
    else:
        print("✗ 部分依赖包安装失败，请手动安装")
        
    print("\n使用示例:")
    print("python cache_usage_examples.py")
    print("python test_cache_performance.py")

if __name__ == "__main__":
    main() 
 