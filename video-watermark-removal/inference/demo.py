import os
import logging
import argparse
from typing import Tuple
import cv2
import torch
import numpy as np

from models.architectures.video_unet import VideoUNet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def tuple_type(s: str) -> Tuple[int, int]:
    try:
        return tuple(map(int, s.split(',')))
    except Exception as e:
        raise argparse.ArgumentTypeError("必须是逗号分隔的整数，例如: 256,256")

def process_single_frame(
    frame: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    frame_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
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

def demo_video(
    video_path: str,
    model: torch.nn.Module,
    device: torch.device,
    frame_size: Tuple[int, int] = (256, 256)
) -> None:
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = process_single_frame(frame, model, device, frame_size)
            processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))
            
            combined = np.hstack((frame, processed_frame))
            cv2.imshow('Original vs Processed', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

def demo_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    frame_size: Tuple[int, int] = (256, 256)
) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"无法读取图像文件: {image_path}")
    
    processed_frame = process_single_frame(frame, model, device, frame_size)
    processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))
    
    combined = np.hstack((frame, processed_frame))
    cv2.imshow('Original vs Processed', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main() -> None:
    parser = argparse.ArgumentParser(description='视频水印去除演示')
    parser.add_argument('--input_path', type=str, required=True, help='输入视频或图像路径')
    parser.add_argument('--model_path', type=str, default='models/pretrained/best_model.pth', help='模型路径')
    parser.add_argument('--frame_size', type=tuple_type, default=(256, 256), help='帧大小 (宽度,高度)')
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.input_path):
            raise FileNotFoundError(f"输入文件不存在: {args.input_path}")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备: {device}')
        
        model = VideoUNet().to(device)
        
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.eval()
            logger.info(f'模型加载完成: {args.model_path}')
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
        
        if args.input_path.endswith(('.mp4', '.avi', '.mov')):
            logger.info('处理视频...')
            demo_video(args.input_path, model, device, args.frame_size)
        else:
            logger.info('处理图像...')
            demo_image(args.input_path, model, device, args.frame_size)
        
        logger.info('演示完成!')
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == '__main__':
    main()
