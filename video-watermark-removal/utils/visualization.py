import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_frame_comparison(original_frame, watermarked_frame, processed_frame, save_path=None):
    """
    可视化原始帧、带水印帧和处理后帧的对比

    Args:
        original_frame: 原始无水印帧
        watermarked_frame: 带水印帧
        processed_frame: 处理后帧
        save_path: 保存路径（可选）
    """
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig)

    # 原始帧
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original (No Watermark)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 带水印帧
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(watermarked_frame, cv2.COLOR_BGR2RGB))
    ax2.set_title('Watermarked', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 处理后帧
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    ax3.set_title('Processed (Watermark Removed)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'对比图已保存到: {save_path}')
    else:
        plt.show()

    plt.close()


def visualize_training_history(train_losses, val_losses, val_psnrs, val_ssims, save_path='training_history.png'):
    """
    可视化训练历史记录

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_psnrs: 验证PSNR列表
        val_ssims: 验证SSIM列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 训练损失
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 验证损失
    axes[0, 1].plot(val_losses, label='Val Loss', color='red', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 验证PSNR
    axes[1, 0].plot(val_psnrs, label='Val PSNR', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1, 0].set_title('Validation PSNR', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # 验证SSIM
    axes[1, 1].plot(val_ssims, label='Val SSIM', color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('SSIM', fontsize=12)
    axes[1, 1].set_title('Validation SSIM', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'训练历史图已保存到: {save_path}')
    plt.close()


def create_video_demo(input_video_path, output_video_path, model, device, frame_size=(256, 256)):
    """
    创建视频演示，对比原始视频、带水印视频和处理后视频

    Args:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
        model: 模型
        device: 设备
        frame_size: 帧大小
    """
    from inference.infer import process_single_frame

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_width = width * 3
    out_height = height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理帧
        processed_frame = process_single_frame(frame, model, device, frame_size)
        processed_frame = cv2.resize(processed_frame, (width, height))

        # 合成带水印帧（这里简化处理，实际应该用真实的水印）
        watermarked_frame = frame.copy()

        # 拼接三个帧
        combined = np.hstack((frame, watermarked_frame, processed_frame))
        out.write(combined)

    cap.release()
    out.release()
    print(f'演示视频已保存到: {output_video_path}')


def visualize_watermark_effect(original_frame, watermark, alpha=0.3, position=(50, 50), save_path=None):
    """
    可视化水印添加效果

    Args:
        original_frame: 原始帧
        watermark: 水印图像
        alpha: 透明度
        position: 水印位置 (x, y)
        save_path: 保存路径（可选）
    """
    h, w = original_frame.shape[:2]
    wh, ww = watermark.shape[:2]

    # 调整水印大小以适应帧
    watermark_resized = cv2.resize(watermark, (min(ww, w//3), min(wh, h//3)))
    wh, ww = watermark_resized.shape[:2]

    # 确保水印在帧内
    x, y = position
    x = min(max(0, x), w - ww)
    y = min(max(0, y), h - wh)

    # 创建带水印的帧
    watermarked = original_frame.copy()

    if watermark_resized.shape[-1] == 4:
        watermark_rgb = watermark_resized[:, :, :3]
        watermark_alpha = watermark_resized[:, :, 3:] / 255.0 * alpha
    else:
        watermark_rgb = watermark_resized
        watermark_alpha = np.ones((wh, ww, 1)) * alpha

    # 混合水印
    for c in range(3):
        watermarked[y:y+wh, x:x+ww, c] = (
            watermarked[y:y+wh, x:x+ww, c] * (1 - watermark_alpha[:, :, 0]) +
            watermark_rgb[:, :, c] * watermark_alpha[:, :, 0]
        )

    # 显示结果
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
        print(f'水印效果对比图已保存到: {save_path}')
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    print("可视化工具模块加载成功")
