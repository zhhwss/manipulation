import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import pickle  # 新增：用于保存聚类中心

# 修复OpenBLAS线程问题
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def save_centers(centers, n_colors, output_dir="kmeans_centers"):
    """保存聚类中心到文件"""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"kmeans_{n_colors}_centers.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(centers, f)
    print(f"已保存 {n_colors} 色聚类中心到: {filename}")

def sample_pixels_from_videos(video_paths, sample_fraction=0.05):
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
                if len(sampled_pixels) < 10:
                    sampled_pixels.append(pixels)
                else:
                    sampled_pixels = [torch.cat(sampled_pixels, dim=0)]
        cap.release()
    return torch.cat(sampled_pixels, dim=0) if sampled_pixels else torch.zeros((0, 3), device=device)

def train_global_kmeans(pixels, n_colors=64):
    """训练K-means模型并保存聚类中心"""
    print(f"训练 {n_colors} 色K-means模型...")
    pixels_np = pixels.cpu().numpy()
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        batch_size=2048,
        n_init=1
    ).fit(pixels_np)
    print("K-means训练完成！")
    return kmeans

def quantize_frame_gpu(frame, kmeans_centers):
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

def process_videos(video_paths, kmeans_centers, output_dir, n_colors):
    """处理视频并保存量化结果"""
    os.makedirs(output_dir, exist_ok=True)
    centers = kmeans_centers.astype(np.float32)
    
    for input_path in tqdm(video_paths, desc=f"量化视频 (n_colors={n_colors})"):
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(
            output_dir,
            os.path.basename(input_path).replace("_color.mp4", f"_kmeans{n_colors}_gpu.mp4")
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
    input_dir = "../lhx/piper_real_cnn/data/20250808/"
    video_paths = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith("_color.mp4")
    ]

    # 采样像素
    pixels = sample_pixels_from_videos(video_paths)
    if len(pixels) == 0:
        raise ValueError("未采样到任何像素！")

    # 训练并保存不同n_colors的模型
    for n_colors in [32, 20, 16]:
        kmeans = train_global_kmeans(pixels, n_colors)
        save_centers(kmeans.cluster_centers_, n_colors)  # 保存聚类中心
        # process_videos(
        #     video_paths,
        #     kmeans.cluster_centers_,
        #     output_dir=f"kmeans_{n_colors}_quantized",
        #     n_colors=n_colors
        # )

    print("所有量化任务完成！")