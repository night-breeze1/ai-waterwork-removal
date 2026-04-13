
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import UNet, Discriminator
from data import WatermarkDataset
from utils import (
    PerceptualLoss, TVLoss, AdversarialLoss,
    calculate_psnr, calculate_ssim, AverageMeter
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Watermark Removal Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    writer = SummaryWriter(config['training']['log_dir'])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print("Loading datasets...")
    train_dataset = WatermarkDataset(
        video_dir=config['data']['train_dir'],
        transform=transform,
        img_size=tuple(config['data']['img_size']),
        max_frames_per_video=config['data']['max_frames_per_video']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    generator = UNet(
        n_channels=config['model']['in_channels'],
        n_classes=config['model']['out_channels'],
        bilinear=config['model']['bilinear']
    ).to(device)
    
    discriminator = Discriminator(in_channels=config['model']['in_channels']).to(device)
    
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        weight_decay=config['training']['weight_decay']
    )
    
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        weight_decay=config['training']['weight_decay']
    )
    
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)
    criterion_tv = TVLoss(weight=config['training']['lambda_tv'])
    criterion_adv = AdversarialLoss(gan_type='lsgan')
    
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    lambda_l1 = config['training']['lambda_l1']
    lambda_perceptual = config['training']['lambda_perceptual']
    lambda_adversarial = config['training']['lambda_adversarial']
    
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        generator.train()
        discriminator.train()
        
        losses_g = AverageMeter()
        losses_d = AverageMeter()
        psnrs = AverageMeter()
        ssims = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["num_epochs"]}')
        for batch_idx, (watermarked, clean) in enumerate(pbar):
            watermarked = watermarked.to(device)
            clean = clean.to(device)
            
            optimizer_d.zero_grad()
            
            fake = generator(watermarked)
            
            pred_real = discriminator(clean)
            pred_fake = discriminator(fake.detach())
            
            loss_d_real = criterion_adv(pred_real, target_is_real=True)
            loss_d_fake = criterion_adv(pred_fake, target_is_real=False)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            
            loss_d.backward()
            optimizer_d.step()
            
            optimizer_g.zero_grad()
            
            pred_fake = discriminator(fake)
            
            loss_l1 = criterion_l1(fake, clean)
            loss_perceptual = criterion_perceptual(fake, clean)
            loss_adv = criterion_adv(pred_fake, target_is_real=True)
            loss_tv = criterion_tv(fake)
            
            loss_g = (
                lambda_l1 * loss_l1 +
                lambda_perceptual * loss_perceptual +
                lambda_adversarial * loss_adv +
                loss_tv
            )
            
            loss_g.backward()
            optimizer_g.step()
            
            losses_g.update(loss_g.item(), watermarked.size(0))
            losses_d.update(loss_d.item(), watermarked.size(0))
            
            for i in range(watermarked.size(0)):
                psnr_val = calculate_psnr(
                    (fake[i] * 0.5 + 0.5).clamp(0, 1),
                    (clean[i] * 0.5 + 0.5).clamp(0, 1)
                )
                ssim_val = calculate_ssim(
                    (fake[i] * 0.5 + 0.5).clamp(0, 1),
                    (clean[i] * 0.5 + 0.5).clamp(0, 1)
                )
                psnrs.update(psnr_val.item())
                ssims.update(ssim_val)
            
            pbar.set_postfix({
                'Loss_G': f'{losses_g.avg:.4f}',
                'Loss_D': f'{losses_d.avg:.4f}',
                'PSNR': f'{psnrs.avg:.2f}',
                'SSIM': f'{ssims.avg:.4f}'
            })
        
        writer.add_scalar('Loss/G', losses_g.avg, epoch)
        writer.add_scalar('Loss/D', losses_d.avg, epoch)
        writer.add_scalar('Metrics/PSNR', psnrs.avg, epoch)
        writer.add_scalar('Metrics/SSIM', ssims.avg, epoch)
        
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['training']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            latest_path = os.path.join(
                config['training']['checkpoint_dir'],
                'latest.pth'
            )
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'config': config
            }, latest_path)
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    main()
