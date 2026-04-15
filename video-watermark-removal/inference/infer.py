import os
import cv2
import torch
import numpy as np
import argparse
from models.architectures.video_unet import VideoUNet

# 自定义tuple类型解析器
def tuple_type(s):
    try:
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("必须是逗号分隔的整数，例如: 256,256")

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
    处理视频

    Args:
        video_path: 视频路径
        model: 模型
        device: 设备
        frame_count: 每段视频的帧数
        frame_size: 帧大小

    Returns:
        处理后的视频帧
    """
    # 输入验证
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 打开视频
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 流式处理视频帧
        processed_frames = []
        frame_buffer = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 调整大小
            frame = cv2.resize(frame, frame_size)
            # 转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 归一化
            frame = frame.astype(np.float32) / 255.0
            
            # 添加到缓冲区
            frame_buffer.append(frame)
            
            # 当缓冲区达到指定帧数时处理
            if len(frame_buffer) == frame_count:
                # 转换为tensor
                batch_frames = torch.from_numpy(np.array(frame_buffer)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
                
                # 前向传播
                with torch.no_grad():
                    output = model(batch_frames)
                
                # 转换回numpy数组
                output = output.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
                output = np.clip(output, 0, 1) * 255
                output = output.astype(np.uint8)
                
                # 转换回BGR格式并添加到结果
                for frame_out in output:
                    frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
                    processed_frames.append(frame_out)
                
                # 清空缓冲区，保留最后一帧作为下一批的开始
                frame_buffer = frame_buffer[-1:]
        
        # 处理剩余的帧
        if frame_buffer:
            # 填充到指定帧数
            while len(frame_buffer) < frame_count:
                frame_buffer.append(frame_buffer[-1])
            
            # 转换为tensor
            batch_frames = torch.from_numpy(np.array(frame_buffer)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            
            # 前向传播
            with torch.no_grad():
                output = model(batch_frames)
            
            # 转换回numpy数组
            output = output.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
            output = np.clip(output, 0, 1) * 255
            output = output.astype(np.uint8)
            
            # 转换回BGR格式并添加到结果
            for frame_out in output:
                frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
                processed_frames.append(frame_out)
    finally:
        if cap is not None:
            cap.release()
    
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
    parser.add_argument('--frame_size', type=tuple_type, default=(256, 256), help='帧大小 (宽度,高度)')
    
    args = parser.parse_args()
    
    # 输入验证
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"视频文件不存在: {args.video_path}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 初始化模型
    model = VideoUNet().to(device)
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print(f'模型加载完成: {args.model_path}')
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")
    
    # 处理视频
    print('处理视频...')
    processed_frames, fps = process_video(args.video_path, model, device, args.frame_count, args.frame_size)
    
    # 保存视频
    save_video(processed_frames, args.output_path, fps)
    
    print('处理完成!')

if __name__ == '__main__':
    main()
