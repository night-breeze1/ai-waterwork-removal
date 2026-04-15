#!/usr/bin/env python3
"""
生成示例训练数据
- 创建合成视频帧
- 创建水印模板
- 生成带水印的训练数据
"""

import os
import cv2
import numpy as np
import random

# 输出目录
VIDEO_DIR = 'raw/videos'
WATERMARK_DIR = 'raw/watermarks'
SYNTHETIC_DIR = 'synthetic'

# 确保目录存在
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(WATERMARK_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

def generate_sample_video_frames(num_frames=10, frame_size=(720, 1280)):
    """生成示例视频帧"""
    frames = []
    for i in range(num_frames):
        # 创建渐变背景
        frame = np.zeros((*frame_size, 3), dtype=np.uint8)
        
        # 添加随机颜色渐变
        for y in range(frame_size[0]):
            for x in range(frame_size[1]):
                r = int(100 + 155 * (x / frame_size[1]))
                g = int(50 + 205 * (y / frame_size[0]))
                b = int(150 + 105 * ((x + y) / (frame_size[0] + frame_size[1])))
                frame[y, x] = [b, g, r]  # OpenCV使用BGR
        
        # 添加一些随机形状
        for _ in range(5):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            x = random.randint(100, frame_size[1] - 100)
            y = random.randint(100, frame_size[0] - 100)
            w = random.randint(50, 200)
            h = random.randint(50, 150)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        
        frames.append(frame)
    return frames

def save_video(frames, output_path, fps=30):
    """保存视频"""
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

def generate_watermarks(num_watermarks=20):
    """生成水印模板"""
    watermarks = []
    
    # 生成B站风格水印
    for i in range(num_watermarks):
        # 创建透明水印
        watermark = np.zeros((100, 200, 4), dtype=np.uint8)
        
        # 填充背景为透明
        watermark[:, :, 3] = 0
        
        # 添加文字
        text = f'bilibili {i+1}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(watermark, text, (10, 60), font, 1.5, (255, 255, 255, 255), 2, cv2.LINE_AA)
        
        # 添加一些装饰元素
        cv2.circle(watermark, (170, 30), 15, (255, 0, 0, 255), -1)
        cv2.rectangle(watermark, (160, 50), (180, 70), (0, 255, 0, 255), -1)
        
        watermarks.append(watermark)
    
    return watermarks

def add_watermark_to_frame(frame, watermark, position=None, alpha=0.3):
    """为帧添加水印"""
    h, w = frame.shape[:2]
    wh, ww = watermark.shape[:2]
    
    if position is None:
        x = random.randint(0, w - ww)
        y = random.randint(0, h - wh)
    else:
        x, y = position
    
    # 确保水印在帧内
    x = min(max(0, x), w - ww)
    y = min(max(0, y), h - wh)
    
    # 复制帧
    watermarked = frame.copy()
    
    # 提取水印的RGB和Alpha通道
    watermark_rgb = watermark[:, :, :3]
    watermark_alpha = watermark[:, :, 3] / 255.0 * alpha
    
    # 混合水印
    for c in range(3):
        watermarked[y:y+wh, x:x+ww, c] = (
            watermarked[y:y+wh, x:x+ww, c] * (1 - watermark_alpha) +
            watermark_rgb[:, :, c] * watermark_alpha
        )
    
    return watermarked

def main():
    print("生成示例视频...")
    
    # 生成10个示例视频
    for i in range(10):
        frames = generate_sample_video_frames()
        output_path = os.path.join(VIDEO_DIR, f'sample_video_{i+1}.mp4')
        save_video(frames, output_path)
        print(f"生成视频: {output_path}")
    
    print("\n生成水印模板...")
    
    # 生成20个水印模板
    watermarks = generate_watermarks()
    for i, watermark in enumerate(watermarks):
        output_path = os.path.join(WATERMARK_DIR, f'watermark_{i+1}.png')
        cv2.imwrite(output_path, watermark)
        print(f"生成水印: {output_path}")
    
    print("\n生成合成训练数据...")
    
    # 为每个视频生成带水印的版本
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    watermark_files = [f for f in os.listdir(WATERMARK_DIR) if f.endswith('.png')]
    
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        # 为每个帧添加水印
        watermarked_frames = []
        for frame in frames:
            watermark_path = os.path.join(WATERMARK_DIR, random.choice(watermark_files))
            watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
            
            # 调整水印大小
            scale = random.uniform(0.5, 1.5)
            new_size = (int(watermark.shape[1] * scale), int(watermark.shape[0] * scale))
            watermark = cv2.resize(watermark, new_size)
            
            # 添加水印
            watermarked = add_watermark_to_frame(frame, watermark, alpha=random.uniform(0.2, 0.5))
            watermarked_frames.append(watermarked)
        
        # 保存带水印的视频
        output_path = os.path.join(SYNTHETIC_DIR, f'watermarked_{video_file}')
        save_video(watermarked_frames, output_path)
        print(f"生成带水印视频: {output_path}")
    
    print("\n数据生成完成！")
    print(f"生成的文件:")
    print(f"- 视频: {len(os.listdir(VIDEO_DIR))} 个")
    print(f"- 水印: {len(os.listdir(WATERMARK_DIR))} 个")
    print(f"- 带水印视频: {len(os.listdir(SYNTHETIC_DIR))} 个")

if __name__ == "__main__":
    main()
