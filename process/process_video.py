import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

# --- 核心调色板设计 ---
def get_final_palette():
    """分区调色板：优先保证6种基础颜色准确映射"""
    # 基础颜色锚点 (必须精确匹配)
    base_colors = {
        'black': [0, 0, 0],
        'white': [255, 255, 255],
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'yellow': [255, 255, 0],
        'purple': [128, 0, 128]
    }
    
    # 扩展色阶 (每个基础颜色扩展3-5个邻近色)
    palette = []
    for color in base_colors.values():
        palette.append(color)
        for i in range(1, 4):
            palette.append(np.clip(color * (1 - i*0.2), 0, 255).astype(int))
    
    # 补充过渡色 (橙/青/粉等)
    extra_colors = [
        [255, 165, 0],  # 橙色
        [0, 255, 255],  # 青色
        [255, 192, 203], # 粉色
        [165, 42, 42],  # 棕色
        [128, 128, 128] # 灰色
    ]
    palette.extend(extra_colors)
    
    return np.unique(np.array(palette), axis=0).astype(np.uint8)

FINAL_PALETTE = get_final_palette()
    
# --- GPU加速量化 ---
def quantize_accurate(frame, palette):
    """分区量化：优先匹配基础颜色"""
    # 将颜色空间划分为基础颜色区和其他颜色区
    base_mask = np.zeros(frame.shape[:2], dtype=bool)
    quantized = np.zeros_like(frame)
    
    # 第一步：精确匹配基础颜色
    for color in ['red', 'green', 'blue', 'yellow', 'black', 'white', 'purple']:
        target = np.array([[255, 0, 0] if color == 'red' else
                          [0, 255, 0] if color == 'green' else
                          [0, 0, 255] if color == 'blue' else
                          [255, 255, 0] if color == 'yellow' else
                          [0, 0, 0] if color == 'black' else
                          [255, 255, 255] if color == 'white' else
                          [128, 0, 128]], dtype=np.uint8)
        
        # 计算颜色距离
        dist = np.linalg.norm(frame - target, axis=2)
        mask = dist < 30  # 宽松阈值捕捉相近色
        quantized[mask] = target
        base_mask |= mask
    
    # 第二步：处理剩余颜色
    remaining_palette = np.array([c for c in palette 
                                if not any((c == [255,0,0]).all() or 
                                          (c == [0,255,0]).all() or
                                          (c == [0,0,255]).all() or
                                          (c == [255,255,0]).all() or
                                          (c == [0,0,0]).all() or
                                          (c == [255,255,255]).all() or
                                          (c == [128,0,128]).all() for c in palette)])
    
    if len(remaining_palette) > 0:
        remaining_frame = frame[~base_mask]
        if len(remaining_frame) > 0:
            distances = torch.cdist(
                torch.from_numpy(remaining_frame).float(), 
                torch.from_numpy(remaining_palette).float()
            )
            _, indices = torch.min(distances, dim=1)
            quantized[~base_mask] = remaining_palette[indices.cpu().numpy()]
    
    return quantized

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

            # 转换颜色空间并量化
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            quantized = quantize_accurate(frame_rgb, FINAL_PALETTE)
            out.write(cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR))
            pbar.update(1)
    
    cap.release()
    out.release()

if __name__ == "__main__":
    # 显示调色板
    import matplotlib.pyplot as plt
    plt.imshow([FINAL_PALETTE], aspect='auto')
    plt.title("Optimized Palette with Base Color Priority")
    plt.show()

    # 处理视频
    input_dir = "../lhx/piper_real_cnn/data/20250808/"
    output_dir = "quantized_output"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith("_color.mp4"):
            process_video(
                os.path.join(input_dir, filename),
                os.path.join(output_dir, filename.replace("_color.mp4", "_quantized.mp4"))
            )