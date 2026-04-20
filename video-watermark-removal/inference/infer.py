import os
import logging
import argparse
from typing import Tuple, List
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

def process_video(
    video_path: str,
    model: torch.nn.Module,
    device: torch.device,
    frame_count: int = 16,
    frame_size: Tuple[int, int] = (256, 256)
) -> Tuple[List[np.ndarray], float]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        processed_frames: List[np.ndarray] = []
        frame_buffer: List[np.ndarray] = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            
            frame_buffer.append(frame)
            
            if len(frame_buffer) == frame_count:
                batch_frames = torch.from_numpy(np.array(frame_buffer)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(batch_frames)
                
                output = output.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
                output = np.clip(output, 0, 1) * 255
                output = output.astype(np.uint8)
                
                for frame_out in output:
                    frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
                    processed_frames.append(frame_out)
                
                frame_buffer = frame_buffer[-1:]
        
        if frame_buffer:
            while len(frame_buffer) < frame_count:
                frame_buffer.append(frame_buffer[-1])
            
            batch_frames = torch.from_numpy(np.array(frame_buffer)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(batch_frames)
            
            output = output.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
            output = np.clip(output, 0, 1) * 255
            output = output.astype(np.uint8)
            
            for frame_out in output:
                frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
                processed_frames.append(frame_out)
    finally:
        if cap is not None:
            cap.release()
    
    processed_frames = processed_frames[:total_frames]
    
    return processed_frames, fps

def save_video(frames: List[np.ndarray], output_path: str, fps: float) -> None:
    if not frames:
        logger.warning('没有处理的帧')
        return
    
    height, width, _ = frames[0].shape
    
    out = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError(f"无法创建输出视频文件: {output_path}")
        
        for frame in frames:
            out.write(frame)
        
        logger.info(f'视频保存到: {output_path}')
    finally:
        if out is not None:
            out.release()

def main() -> None:
    parser = argparse.ArgumentParser(description='视频水印去除推理')
    parser.add_argument('--video_path', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='输出视频路径')
    parser.add_argument('--model_path', type=str, default='models/pretrained/best_model.pth', help='模型路径')
    parser.add_argument('--frame_count', type=int, default=16, help='每段视频的帧数')
    parser.add_argument('--frame_size', type=tuple_type, default=(256, 256), help='帧大小 (宽度,高度)')
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"视频文件不存在: {args.video_path}")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
        if args.frame_count <= 0:
            raise ValueError("帧数必须大于0")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备: {device}')
        
        model = VideoUNet().to(device)
        
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.eval()
            logger.info(f'模型加载完成: {args.model_path}')
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
        
        logger.info('处理视频...')
        processed_frames, fps = process_video(args.video_path, model, device, args.frame_count, args.frame_size)
        
        save_video(processed_frames, args.output_path, fps)
        
        logger.info('处理完成!')
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == '__main__':
    main()
