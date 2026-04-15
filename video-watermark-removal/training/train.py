import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from tqdm import tqdm
import json

# 自定义tuple类型解析器
def tuple_type(s):
    try:
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("必须是逗号分隔的整数，例如: 256,256")

from models.architectures.video_unet import VideoUNet
from utils.data import create_dataloader
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import visualize_training_history

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
    
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (watermarked_frames, target_frames) in enumerate(tqdm(dataloader)):
        # 移动数据到设备
        watermarked_frames = watermarked_frames.to(device)
        target_frames = target_frames.to(device)
        
        # 前向传播
        output = model(watermarked_frames)
        
        # 计算损失
        loss = criterion(output, target_frames)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        平均损失、PSNR、SSIM
    """
    model.eval()
    total_loss = 0.0
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for batch_idx, (watermarked_frames, target_frames) in enumerate(tqdm(dataloader)):
            # 移动数据到设备
            watermarked_frames = watermarked_frames.to(device)
            target_frames = target_frames.to(device)
            
            # 前向传播
            output = model(watermarked_frames)
            
            # 计算损失
            loss = criterion(output, target_frames)
            total_loss += loss.item()
            
            # 计算评估指标
            psnr = calculate_psnr(output, target_frames)
            ssim_val = calculate_ssim(output, target_frames)
            psnr_values.append(psnr)
            ssim_values.append(ssim_val)
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    
    return avg_loss, avg_psnr, avg_ssim

def main():
    # 解析命令行参数
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
    
    # 创建模型保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 初始化模型
    model = VideoUNet().to(device)
    print(f'模型初始化完成')
    
    # 创建数据加载器
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
    
    print(f'数据加载器创建完成，训练集大小: {len(train_dataloader.dataset)}')
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 定义学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    
    # 训练循环
    best_psnr = 0.0
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 50)
        
        # 训练
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        train_losses.append(train_loss)
        print(f'Train Loss: {train_loss:.4f}')
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_psnr, val_ssim = validate(model, val_dataloader, criterion, device)
            val_losses.append(val_loss)
            val_psnrs.append(val_psnr)
            val_ssims.append(val_ssim)
            print(f'Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}')
            
            # 保存最佳模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
                print(f'保存最佳模型，PSNR: {best_psnr:.2f}')
        else:
            # 填充未验证的epoch，保持列表长度一致
            val_losses.append(None)
            val_psnrs.append(None)
            val_ssims.append(None)
        
        # 定期保存模型
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, f'model_epoch_{epoch+1}.pth'))
            print(f'保存模型到: {os.path.join(args.model_dir, f"model_epoch_{epoch+1}.pth")}')
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': [float(x) if x is not None else None for x in val_losses],
        'val_psnrs': [float(x) if x is not None else None for x in val_psnrs],
        'val_ssims': [float(x) if x is not None else None for x in val_ssims]
    }
    
    with open(os.path.join(args.model_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    print(f'训练历史已保存到: {os.path.join(args.model_dir, "training_history.json")}')
    
    # 可视化训练历史
    visualize_training_history(
        train_losses,
        [x for x in val_losses if x is not None],
        [x for x in val_psnrs if x is not None],
        [x for x in val_ssims if x is not None],
        save_path=os.path.join(args.model_dir, 'training_history.png')
    )
    
    print('\n训练完成!')

if __name__ == '__main__':
    main()
