import os
import logging
from typing import Optional, Tuple, List
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gc
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def visualize_frame_comparison(
    original_frame: np.ndarray,
    watermarked_frame: np.ndarray,
    processed_frame: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original (No Watermark)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(watermarked_frame, cv2.COLOR_BGR2RGB))
    ax2.set_title('Watermarked', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    ax3.set_title('Processed (Watermark Removed)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'对比图已保存到: {save_path}')
    else:
        plt.show()

    plt.close()
    gc.collect()

def visualize_training_history(
    train_losses: List[float],
    val_losses: List[float],
    val_psnrs: List[float],
    val_ssims: List[float],
    save_path: str = 'training_history.png'
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(val_losses, label='Val Loss', color='red', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(val_psnrs, label='Val PSNR', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1, 0].set_title('Validation PSNR', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(val_ssims, label='Val SSIM', color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('SSIM', fontsize=12)
    axes[1, 1].set_title('Validation SSIM', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f'训练历史图已保存到: {save_path}')
    plt.close()
    gc.collect()

def create_video_demo(
    input_video_path: str,
    output_video_path: str,
    model: torch.nn.Module,
    device: torch.device,
    frame_size: Tuple[int, int] = (256, 256)
) -> None:
    from inference.infer import process_single_frame

    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {input_video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_width = width * 3
        out_height = height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))
        if not out.isOpened():
            raise RuntimeError(f"无法创建输出视频文件: {output_video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_single_frame(frame, model, device, frame_size)
            processed_frame = cv2.resize(processed_frame, (width, height))

            watermarked_frame = frame.copy()

            combined = np.hstack((frame, watermarked_frame, processed_frame))
            out.write(combined)
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
    logger.info(f'演示视频已保存到: {output_video_path}')

def visualize_watermark_effect(
    original_frame: np.ndarray,
    watermark: np.ndarray,
    alpha: float = 0.3,
    position: Tuple[int, int] = (50, 50),
    save_path: Optional[str] = None
) -> None:
    h, w = original_frame.shape[:2]
    wh, ww = watermark.shape[:2]

    watermark_resized = cv2.resize(watermark, (min(ww, w//3), min(wh, h//3)))
    wh, ww = watermark_resized.shape[:2]

    x, y = position
    x = min(max(0, x), w - ww)
    y = min(max(0, y), h - wh)

    watermarked = original_frame.copy()

    if watermark_resized.shape[-1] == 4:
        watermark_rgb = watermark_resized[:, :, :3]
        watermark_alpha = watermark_resized[:, :, 3:] / 255.0 * alpha
    else:
        watermark_rgb = watermark_resized
        watermark_alpha = np.ones((wh, ww, 1)) * alpha

    for c in range(3):
        watermarked[y:y+wh, x:x+ww, c] = (
            watermarked[y:y+wh, x:x+ww, c] * (1 - watermark_alpha[:, :, 0]) +
            watermark_rgb[:, :, c] * watermark_alpha[:, :, 0]
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Watermarked', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'水印效果对比图已保存到: {save_path}')
    else:
        plt.show()

    plt.close()
    gc.collect()

if __name__ == "__main__":
    logger.info("可视化工具模块加载成功")
