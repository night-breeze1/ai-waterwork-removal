import os
import cv2
import torch
import numpy as np
import argparse
from models.architectures.video_unet import VideoUNet

def process_single_frame(frame, model, device, frame_size=(256, 256)):
    """
    处理单帧图像

    Args:
        frame: 输入帧
        model: 模型
        device: 设备
        frame_size: 帧大小

    Returns:
        处理后的帧
    """
    resized_frame = cv2.resize(frame, frame_size)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    normalized_frame = rgb_frame.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(normalized_frame).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    output = output.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output, 0, 1) * 255
    output = output.astype(np.uint8)
    output_frame = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    return output_frame

def process_video(video_path, model, device, frame_count=16, frame_size=(256, 256)):
    """
    处理视频文件，去除水印
    
    Args:
        video_path: 视频路径
        model: 模型
        device: 设备
        frame_count: 每段视频的帧数
        frame_size: 帧大小
    
    Returns:
        处理后的视频帧
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 读取视频帧
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 调整大小
        frame = cv2.resize(frame, frame_size)
        # 转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    # 转换为numpy数组
    frames = np.array(frames, dtype=np.float32) / 255.0
    
    # 分批次处理
    processed_frames = []
    for i in range(0, len(frames), frame_count):
        # 获取当前批次的帧
        batch_frames = frames[i:i+frame_count]
        
        # 确保批次大小正确
        if len(batch_frames) < frame_count:
            # 填充最后一帧
            while len(batch_frames) < frame_count:
                batch_frames = np.append(batch_frames, [batch_frames[-1]], axis=0)
        
        # 转换为tensor
        batch_frames = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        
        # 前向传播
        with torch.no_grad():
            output = model(batch_frames)
        
        # 转换回numpy数组
        output = output.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        output = np.clip(output, 0, 1) * 255
        output = output.astype(np.uint8)
        
        # 转换回BGR格式
        for frame in output:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed_frames.append(frame)
    
    # 截取实际帧数
    processed_frames = processed_frames[:total_frames]
    
    return processed_frames, fps

def save_video(frames, output_path, fps):
    """
    保存处理后的视频
    
    Args:
        frames: 处理后的帧
        output_path: 输出路径
        fps: 帧率
    """
    if not frames:
        print('没有处理的帧')
        return
    
    # 获取帧大小
    height, width, _ = frames[0].shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 写入帧
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f'视频保存到: {output_path}')

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='视频水印去除推理')
    parser.add_argument('--video_path', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='输出视频路径')
    parser.add_argument('--model_path', type=str, default='models/pretrained/best_model.pth', help='模型路径')
    parser.add_argument('--frame_count', type=int, default=16, help='每段视频的帧数')
    parser.add_argument('--frame_size', type=tuple, default=(256, 256), help='帧大小')
    
    args = parser.parse_args()
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 初始化模型
    model = VideoUNet().to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f'模型加载完成: {args.model_path}')
    
    # 处理视频
    print('处理视频...')
    processed_frames, fps = process_video(args.video_path, model, device, args.frame_count, args.frame_size)
    
    # 保存视频
    save_video(processed_frames, args.output_path, fps)
    
    print('处理完成!')

if __name__ == '__main__':
    main()
