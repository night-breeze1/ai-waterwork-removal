import time
import logging
from typing import Tuple
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.float()
    target = target.float()
    
    mse = F.mse_loss(pred, target, reduction='mean')
    
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    
    return psnr.item()

def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    ssim_values = []
    for i in range(pred_np.shape[0]):
        for j in range(pred_np.shape[1]):
            pred_frame = pred_np[i, j].transpose(1, 2, 0)
            target_frame = target_np[i, j].transpose(1, 2, 0)
            
            ssim_val = ssim(pred_frame, target_frame, channel_axis=2, data_range=1.0)
            ssim_values.append(ssim_val)
    
    return float(np.mean(ssim_values))

def calculate_fps(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int, int] = (1, 16, 3, 256, 256)
) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.randn(input_size).to(device)
    
    for _ in range(10):
        with torch.no_grad():
            model(input_tensor)
    
    start_time = time.time()
    iterations = 100
    
    with torch.no_grad():
        for _ in range(iterations):
            model(input_tensor)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = (iterations * input_size[1]) / elapsed_time
    
    return fps

if __name__ == "__main__":
    pred = torch.randn(1, 16, 3, 256, 256)
    target = torch.randn(1, 16, 3, 256, 256)
    
    psnr = calculate_psnr(pred, target)
    ssim_val = calculate_ssim(pred, target)
    
    logger.info(f"PSNR: {psnr:.2f} dB")
    logger.info(f"SSIM: {ssim_val:.4f}")
