
import os
import argparse
import yaml
import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from models import UNet


def parse_args():
    parser = argparse.ArgumentParser(description='Video Watermark Removal Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input video or directory of videos')
    parser.add_argument('--output', type=str, default='./output',
                        help='Path to output directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, config, device):
    generator = UNet(
        n_channels=config['model']['in_channels'],
        n_classes=config['model']['out_channels'],
        bilinear=config['model']['bilinear']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    generator.to(device)
    
    return generator


def process_video(video_path, output_path, generator, device, transform, img_size, batch_size=4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames = []
    frame_count = 0
    
    pbar = tqdm(total=total_frames, desc=f'Processing {os.path.basename(video_path)}')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        original_size = frame.shape[:2]
        frame_resized = cv2.resize(frame, img_size)
        frames.append((frame, frame_resized))
        frame_count += 1
        
        if len(frames) >= batch_size or frame_count == total_frames:
            batch_frames_resized = [f[1] for f in frames]
            batch_tensors = []
            
            for f in batch_frames_resized:
                f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                tensor = transform(f_rgb).unsqueeze(0)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
            
            with torch.no_grad():
                output_tensor = generator(batch_tensor)
            
            for i, (original_frame, _) in enumerate(frames):
                output_img = output_tensor[i].cpu().permute(1, 2, 0).numpy()
                output_img = (output_img * 0.5 + 0.5) * 255
                output_img = np.clip(output_img, 0, 255).astype(np.uint8)
                output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                output_img = cv2.resize(output_img, (width, height))
                
                out.write(output_img)
            
            frames = []
            pbar.update(len(batch_frames_resized))
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"Video saved to: {output_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading model from {args.checkpoint}")
    generator = load_model(args.checkpoint, config, device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_size = tuple(config['data']['img_size'])
    
    if os.path.isfile(args.input):
        video_files = [args.input]
    else:
        video_files = []
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(root, file))
    
    if not video_files:
        print("No video files found!")
        return
    
    print(f"Found {len(video_files)} video(s) to process")
    
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        name, ext = os.path.splitext(video_name)
        output_name = f"{name}_no_watermark{ext}"
        output_path = os.path.join(args.output, output_name)
        
        process_video(
            video_path=video_path,
            output_path=output_path,
            generator=generator,
            device=device,
            transform=transform,
            img_size=img_size,
            batch_size=args.batch_size
        )


if __name__ == '__main__':
    main()
