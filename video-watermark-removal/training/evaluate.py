import torch
import argparse
from models.architectures.video_unet import VideoUNet
from utils.data import create_dataloader
from utils.metrics import calculate_psnr, calculate_ssim, calculate_fps

# 自定义tuple类型解析器
def tuple_type(s):
    try:
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("必须是逗号分隔的整数，例如: 256,256")

def evaluate_model(model, dataloader, device):
    """
    评估模型性能
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        PSNR、SSIM、FPS
    """
    model.eval()
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for batch_idx, (watermarked_frames, target_frames) in enumerate(dataloader):
            # 移动数据到设备
            watermarked_frames = watermarked_frames.to(device)
            target_frames = target_frames.to(device)
            
            # 前向传播
            output = model(watermarked_frames)
            
            # 计算评估指标
            psnr = calculate_psnr(output, target_frames)
            ssim_val = calculate_ssim(output, target_frames)
            psnr_values.append(psnr)
            ssim_values.append(ssim_val)
    
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    
    # 计算FPS
    fps = calculate_fps(model)
    
    return avg_psnr, avg_ssim, fps

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估视频水印去除模型')
    parser.add_argument('--video_dir', type=str, default='data/raw/videos', help='无水印视频目录')
    parser.add_argument('--watermark_dir', type=str, default='data/raw/watermarks', help='水印图像目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--frame_count', type=int, default=16, help='每段视频的帧数')
    parser.add_argument('--frame_size', type=tuple_type, default=(256, 256), help='帧大小 (宽度,高度)')
    parser.add_argument('--model_path', type=str, default='models/pretrained/best_model.pth', help='模型路径')
    
    args = parser.parse_args()
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 初始化模型
    model = VideoUNet().to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f'模型加载完成: {args.model_path}')
    
    # 创建数据加载器
    dataloader = create_dataloader(
        args.video_dir, 
        args.watermark_dir, 
        batch_size=args.batch_size, 
        frame_count=args.frame_count, 
        frame_size=args.frame_size,
        shuffle=False
    )
    
    print(f'数据加载器创建完成，测试集大小: {len(dataloader.dataset)}')
    
    # 评估模型
    psnr, ssim_val, fps = evaluate_model(model, dataloader, device)
    
    print('\n评估结果:')
    print(f'PSNR: {psnr:.2f} dB')
    print(f'SSIM: {ssim_val:.4f}')
    print(f'FPS: {fps:.2f}')

if __name__ == '__main__':
    main()
