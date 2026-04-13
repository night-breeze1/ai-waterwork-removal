import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_psnr(pred, target):
    """
    计算峰值信噪比（PSNR）
    
    Args:
        pred: 预测结果 (batch, frames, channels, height, width)
        target: 目标结果 (batch, frames, channels, height, width)
    
    Returns:
        PSNR值
    """
    # 确保输入是浮点型
    pred = pred.float()
    target = target.float()
    
    # 计算MSE
    mse = F.mse_loss(pred, target, reduction='mean')
    
    # 计算PSNR
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # 因为我们已经将像素值归一化到[0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    
    return psnr.item()

def calculate_ssim(pred, target):
    """
    计算结构相似性指数（SSIM）
    
    Args:
        pred: 预测结果 (batch, frames, channels, height, width)
        target: 目标结果 (batch, frames, channels, height, width)
    
    Returns:
        SSIM值
    """
    # 转换为numpy数组
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # 计算每帧的SSIM
    ssim_values = []
    for i in range(pred_np.shape[0]):  # 批次
        for j in range(pred_np.shape[1]):  # 帧
            # 转换为 (height, width, channels) 格式
            pred_frame = pred_np[i, j].transpose(1, 2, 0)
            target_frame = target_np[i, j].transpose(1, 2, 0)
            
            # 计算SSIM
            ssim_val = ssim(pred_frame, target_frame, channel_axis=2, data_range=1.0)
            ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

def calculate_fps(model, input_size=(1, 16, 3, 256, 256)):
    """
    计算模型的推理速度（FPS）
    
    Args:
        model: 模型
        input_size: 输入大小 (batch, frames, channels, height, width)
    
    Returns:
        FPS值
    """
    # 创建随机输入
    input = torch.randn(input_size).cuda() if torch.cuda.is_available() else torch.randn(input_size)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            model(input)
    
    # 测试
    import time
    start_time = time.time()
    iterations = 100
    
    with torch.no_grad():
        for _ in range(iterations):
            model(input)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = (iterations * input_size[1]) / elapsed_time  # 总帧数 / 总时间
    
    return fps

if __name__ == "__main__":
    # 测试评估指标
    pred = torch.randn(1, 16, 3, 256, 256)
    target = torch.randn(1, 16, 3, 256, 256)
    
    psnr = calculate_psnr(pred, target)
    ssim_val = calculate_ssim(pred, target)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
