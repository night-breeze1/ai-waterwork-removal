import os
import logging
import argparse
import json
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.architectures.video_unet import VideoUNet
from utils.data import create_dataloader
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import visualize_training_history

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

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    
    for batch_idx, (watermarked_frames, target_frames) in enumerate(tqdm(dataloader)):
        watermarked_frames = watermarked_frames.to(device)
        target_frames = target_frames.to(device)
        
        output = model(watermarked_frames)
        loss = criterion(output, target_frames)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for batch_idx, (watermarked_frames, target_frames) in enumerate(tqdm(dataloader)):
            watermarked_frames = watermarked_frames.to(device)
            target_frames = target_frames.to(device)
            
            output = model(watermarked_frames)
            loss = criterion(output, target_frames)
            total_loss += loss.item()
            
            psnr = calculate_psnr(output, target_frames)
            ssim_val = calculate_ssim(output, target_frames)
            psnr_values.append(psnr)
            ssim_values.append(ssim_val)
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    
    return avg_loss, avg_psnr, avg_ssim

def main() -> None:
    parser = argparse.ArgumentParser(description='训练视频水印去除模型')
    parser.add_argument('--video_dir', type=str, default='data/raw/videos', help='无水印视频目录')
    parser.add_argument('--watermark_dir', type=str, default='data/raw/watermarks', help='水印图像目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--frame_count', type=int, default=16, help='每段视频的帧数')
    parser.add_argument('--frame_size', type=tuple_type, default=(256, 256), help='帧大小 (宽度,高度)')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--save_interval', type=int, default=20, help='模型保存间隔')
    parser.add_argument('--val_interval', type=int, default=5, help='验证间隔')
    parser.add_argument('--model_dir', type=str, default='models/pretrained', help='模型保存目录')
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.video_dir):
            raise FileNotFoundError(f"视频目录不存在: {args.video_dir}")
        if not os.path.exists(args.watermark_dir):
            raise FileNotFoundError(f"水印目录不存在: {args.watermark_dir}")
        if args.batch_size <= 0:
            raise ValueError("批次大小必须大于0")
        if args.frame_count <= 0:
            raise ValueError("帧数必须大于0")
        if args.epochs <= 0:
            raise ValueError("训练轮次必须大于0")
        if args.lr <= 0:
            raise ValueError("学习率必须大于0")
        
        os.makedirs(args.model_dir, exist_ok=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备: {device}')
        
        model = VideoUNet().to(device)
        logger.info('模型初始化完成')
        
        train_dataloader = create_dataloader(
            args.video_dir,
            args.watermark_dir,
            batch_size=args.batch_size,
            frame_count=args.frame_count,
            frame_size=args.frame_size,
            shuffle=True
        )
        
        val_dataloader = create_dataloader(
            args.video_dir,
            args.watermark_dir,
            batch_size=args.batch_size,
            frame_count=args.frame_count,
            frame_size=args.frame_size,
            shuffle=False
        )
        
        logger.info(f'数据加载器创建完成，训练集大小: {len(train_dataloader.dataset)}')
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        train_losses: List[float] = []
        val_losses: List[Optional[float]] = []
        val_psnrs: List[Optional[float]] = []
        val_ssims: List[Optional[float]] = []
        
        best_psnr = 0.0
        for epoch in range(args.epochs):
            logger.info(f'\nEpoch {epoch+1}/{args.epochs}')
            logger.info('-' * 50)
            
            train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
            train_losses.append(train_loss)
            logger.info(f'Train Loss: {train_loss:.4f}')
            
            scheduler.step()
            
            if (epoch + 1) % args.val_interval == 0:
                val_loss, val_psnr, val_ssim = validate(model, val_dataloader, criterion, device)
                val_losses.append(val_loss)
                val_psnrs.append(val_psnr)
                val_ssims.append(val_ssim)
                logger.info(f'Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}')
                
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
                    logger.info(f'保存最佳模型，PSNR: {best_psnr:.2f}')
            else:
                val_losses.append(None)
                val_psnrs.append(None)
                val_ssims.append(None)
            
            if (epoch + 1) % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.model_dir, f'model_epoch_{epoch+1}.pth'))
                logger.info(f'保存模型到: {os.path.join(args.model_dir, f"model_epoch_{epoch+1}.pth")}')
        
        history = {
            'train_losses': train_losses,
            'val_losses': [float(x) if x is not None else None for x in val_losses],
            'val_psnrs': [float(x) if x is not None else None for x in val_psnrs],
            'val_ssims': [float(x) if x is not None else None for x in val_ssims]
        }
        
        with open(os.path.join(args.model_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
        logger.info(f'训练历史已保存到: {os.path.join(args.model_dir, "training_history.json")}')
        
        visualize_training_history(
            train_losses,
            [x for x in val_losses if x is not None],
            [x for x in val_psnrs if x is not None],
            [x for x in val_ssims if x is not None],
            save_path=os.path.join(args.model_dir, 'training_history.png')
        )
        
        logger.info('\n训练完成!')
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == '__main__':
    main()
