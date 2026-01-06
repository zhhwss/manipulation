import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import os

# 修复OpenBLAS线程问题
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # 限制OpenBLAS线程数

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def sample_pixels_from_videos(video_paths, sample_fraction=0.05):  # 降低采样率减少内存压力
    """从所有视频中采样像素（GPU优化版）"""
    sampled_pixels = []
    for path in tqdm(video_paths, desc="采样视频像素"):
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if np.random.rand() < sample_fraction:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pixels = torch.from_numpy(frame_rgb).float().to(device).view(-1, 3)
                if len(sampled_pixels) < 10:  # 分批处理避免内存爆炸
                    sampled_pixels.append(pixels)
                else:
                    sampled_pixels = [torch.cat(sampled_pixels, dim=0)]  # 合并已采样的
        cap.release()
    return torch.cat(sampled_pixels, dim=0) if sampled_pixels else torch.zeros((0, 3), device=device)

def train_global_kmeans(pixels, n_colors=64):
    """在GPU上训练K-means模型"""
    print("训练K-means模型...")
    pixels_np = pixels.cpu().numpy()
    
    # 使用更小的batch_size和n_init
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        batch_size=2048,  # 减小batch_size
        n_init=1          # 减少初始化次数
    ).fit(pixels_np)
    print("K-means训练完成！")
    return kmeans

def quantize_frame_gpu(frame, kmeans_centers):
    """GPU加速的量化"""
    h, w = frame.shape[:2]
    frame_tensor = torch.from_numpy(frame).float().to(device).view(-1, 3)
    centers_tensor = torch.from_numpy(kmeans_centers).float().to(device)
    
    # 分块处理避免显存不足
    chunk_size = 100000  # 每块10万像素
    quantized = torch.zeros_like(frame_tensor)
    for i in range(0, len(frame_tensor), chunk_size):
        chunk = frame_tensor[i:i+chunk_size]
        distances = torch.cdist(chunk, centers_tensor, p=2)
        labels = torch.argmin(distances, dim=1)
        quantized[i:i+chunk_size] = centers_tensor[labels]
    
    return quantized.view(h, w, 3).byte().cpu().numpy()

def process_videos(video_paths, kmeans_centers, output_dir):
    """处理所有视频（显存优化版）"""
    os.makedirs(output_dir, exist_ok=True)
    centers = kmeans_centers.astype(np.float32)
    
    for input_path in tqdm(video_paths, desc="量化视频"):
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(
            output_dir,
            os.path.basename(input_path).replace("_color.mp4", "_kmeans64_gpu.mp4")
        )
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            quantized = quantize_frame_gpu(frame_rgb, centers)
            out.write(cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR))
        
        cap.release()
        out.release()

if __name__ == "__main__":
    # 配置路径
    input_dir = "../lhx/piper_real_cnn/data/20250808/"
    output_dir = "gpu_kmeans_quantized_fixed"
    video_paths = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith("_color.mp4")
    ]

    # 采样像素并训练K-means
    pixels = sample_pixels_from_videos(video_paths)
    if len(pixels) == 0:
        raise ValueError("未采样到任何像素！请检查视频路径或采样率")
    kmeans = train_global_kmeans(pixels, n_colors=16)

    # 量化所有视频
    process_videos(video_paths, kmeans.cluster_centers_, output_dir)
    print(f"量化完成！结果保存在: {output_dir}")