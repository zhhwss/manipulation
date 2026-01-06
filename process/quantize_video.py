import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 严格红绿调色板生成 ---
def get_red_green_palette():
    """确保红色和绿色完全准确，其他颜色平滑过渡"""
    # 基础颜色（精确值）
    core_colors = np.array([
        [255, 0, 0],      # 纯红 (R=255, G=0, B=0)
        [0, 255, 0],      # 纯绿 (R=0, G=255, B=0)
        [255, 255, 255],  # 白
        [128, 0, 128],    # 紫
        [255, 255, 0],    # 黄
        [255, 165, 0],    # 橙
        [0, 0, 0],        # 黑
        [255, 200, 150],  # 浅肉色
        [210, 180, 140],  # 深肉色
        [0, 0, 255],      # 蓝
        [128, 128, 128]   # 灰
    ], dtype=np.uint8)

    # 为红、绿生成保护性渐变色（避免邻近色干扰）
    red_green_protection = np.array([
        [255, 50, 50],    # 防红-橙过渡
        [50, 255, 50],    # 防绿-青过渡
    ])
    
    return np.unique(np.vstack([core_colors, red_green_protection]), axis=0)

# --- 强制红绿优先的量化 ---
def quantize_red_green_priority(frame, palette):
    """强制红色和绿色优先精确匹配"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    
    # 转为PyTorch Tensor
    frame_tensor = torch.from_numpy(frame_rgb).float().to(device)
    palette_tensor = torch.from_numpy(palette).float().to(device)
    
    # 计算所有像素到调色板的距离
    distances = torch.cdist(
        frame_tensor.view(-1, 3), 
        palette_tensor, 
        p=2
    )
    
    # 红绿强制匹配（严格阈值）
    red_mask = (frame_tensor[..., 0] > 200) & (frame_tensor[..., 1] < 50) & (frame_tensor[..., 2] < 50)
    green_mask = (frame_tensor[..., 1] > 200) & (frame_tensor[..., 0] < 50) & (frame_tensor[..., 2] < 50)
    
    # 红色直接映射到调色板中的纯红（索引0）
    if red_mask.any():
        distances[red_mask.view(-1)] = float('inf')
        distances[red_mask.view(-1), 0] = 0  # 纯红
    
    # 绿色直接映射到调色板中的纯绿（索引1）
    if green_mask.any():
        distances[green_mask.view(-1)] = float('inf')
        distances[green_mask.view(-1), 1] = 0  # 纯绿
    
    # 其他颜色正常匹配
    _, indices = torch.min(distances, dim=1)
    quantized_rgb = palette_tensor[indices].view(h, w, 3)
    
    return quantized_rgb.byte().cpu().numpy()

# --- 视频处理流程 ---
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with tqdm(desc=f"Processing {os.path.basename(input_path)}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            quantized = quantize_red_green_priority(frame, RED_GREEN_PALETTE)
            out.write(cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR))
            pbar.update(1)
    
    cap.release()
    out.release()

if __name__ == "__main__":
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成调色板
    RED_GREEN_PALETTE = get_red_green_palette()
    
    # 显示调色板
    plt.figure(figsize=(15, 2))
    plt.imshow([RED_GREEN_PALETTE], aspect='auto')
    plt.title("Red/Green Priority Palette")
    plt.axis('off')
    plt.show()

    # 处理视频
    input_dir = "../lhx/piper_real_cnn/data/20250808/"
    output_dir = "quantized_output_red_green"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith("_color.mp4"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace("_color.mp4", "_quantized_rg.mp4"))
            process_video(input_path, output_path)