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
    # 调整大小
    resized_frame = cv2.resize(frame, frame_size)
    # 转换为RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # 归一化
    normalized_frame = rgb_frame.astype(np.float32) / 255.0
    
    # 转换为tensor并添加批次和时间维度
    input_tensor = torch.from_numpy(normalized_frame).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    # 转换回numpy数组
    output = output.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output, 0, 1) * 255
    output = output.astype(np.uint8)
    
    # 转换回BGR格式
    output_frame = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    return output_frame

def demo_video(video_path, model, device, frame_size=(256, 256)):
    """
    演示视频处理效果
    
    Args:
        video_path: 视频路径
        model: 模型
        device: 设备
        frame_size: 帧大小
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        processed_frame = process_single_frame(frame, model, device, frame_size)
        
        # 调整处理后帧的大小以匹配原始帧
        processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))
        
        # 显示原始帧和处理后帧
        combined = np.hstack((frame, processed_frame))
        cv2.imshow('Original vs Processed', combined)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def demo_image(image_path, model, device, frame_size=(256, 256)):
    """
    演示图像处理效果
    
    Args:
        image_path: 图像路径
        model: 模型
        device: 设备
        frame_size: 帧大小
    """
    # 读取图像
    frame = cv2.imread(image_path)
    
    # 处理帧
    processed_frame = process_single_frame(frame, model, device, frame_size)
    
    # 调整处理后帧的大小以匹配原始帧
    processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))
    
    # 显示原始图像和处理后图像
    combined = np.hstack((frame, processed_frame))
    cv2.imshow('Original vs Processed', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='视频水印去除演示')
    parser.add_argument('--input_path', type=str, required=True, help='输入视频或图像路径')
    parser.add_argument('--model_path', type=str, default='models/pretrained/best_model.pth', help='模型路径')
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
    
    # 检查输入类型
    if args.input_path.endswith(('.mp4', '.avi', '.mov')):
        # 处理视频
        print('处理视频...')
        demo_video(args.input_path, model, device, args.frame_size)
    else:
        # 处理图像
        print('处理图像...')
        demo_image(args.input_path, model, device, args.frame_size)
    
    print('演示完成!')

if __name__ == '__main__':
    main()
