import os
import logging
import argparse
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader

from models.architectures.video_unet import VideoUNet
from utils.data import create_dataloader
from utils.metrics import calculate_psnr, calculate_ssim, calculate_fps

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

def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    psnr_values = []
    ssim_values = []
    
    try:
        with torch.no_grad():
            for batch_idx, (watermarked_frames, target_frames) in enumerate(dataloader):
                watermarked_frames = watermarked_frames.to(device)
                target_frames = target_frames.to(device)
                
                output = model(watermarked_frames)
                
                psnr = calculate_psnr(output, target_frames)
                ssim_val = calculate_ssim(output, target_frames)
                psnr_values.append(psnr)
                ssim_values.append(ssim_val)
        
        if not psnr_values:
            raise ValueError("没有有效的评估批次")
        
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        fps = calculate_fps(model)
        
        return avg_psnr, avg_ssim, fps
    except Exception as e:
        logger.error(f"评估过程出错: {e}")
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description='评估视频水印去除模型')
    parser.add_argument('--video_dir', type=str, default='data/raw/videos', help='无水印视频目录')
    parser.add_argument('--watermark_dir', type=str, default='data/raw/watermarks', help='水印图像目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--frame_count', type=int, default=16, help='每段视频的帧数')
    parser.add_argument('--frame_size', type=tuple_type, default=(256, 256), help='帧大小 (宽度,高度)')
    parser.add_argument('--model_path', type=str, default='models/pretrained/best_model.pth', help='模型路径')
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.video_dir):
            raise FileNotFoundError(f"视频目录不存在: {args.video_dir}")
        if not os.path.exists(args.watermark_dir):
            raise FileNotFoundError(f"水印目录不存在: {args.watermark_dir}")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
        if args.batch_size <= 0:
            raise ValueError("批次大小必须大于0")
        if args.frame_count <= 0:
            raise ValueError("帧数必须大于0")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备: {device}')
        
        model = VideoUNet().to(device)
        
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            logger.info(f'模型加载完成: {args.model_path}')
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
        
        dataloader = create_dataloader(
            args.video_dir,
            args.watermark_dir,
            batch_size=args.batch_size,
            frame_count=args.frame_count,
            frame_size=args.frame_size,
            shuffle=False
        )
        
        logger.info(f'数据加载器创建完成，测试集大小: {len(dataloader.dataset)}')
        
        psnr, ssim_val, fps = evaluate_model(model, dataloader, device)
        
        logger.info('\n评估结果:')
        logger.info(f'PSNR: {psnr:.2f} dB')
        logger.info(f'SSIM: {ssim_val:.4f}')
        logger.info(f'FPS: {fps:.2f}')
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == '__main__':
    main()
