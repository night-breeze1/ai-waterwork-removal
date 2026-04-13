
import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class WatermarkDataset(Dataset):
    def __init__(self, video_dir, transform=None, img_size=(256, 256), 
                 max_frames_per_video=10, watermark_types=None):
        self.video_dir = video_dir
        self.transform = transform
        self.img_size = img_size
        self.max_frames_per_video = max_frames_per_video
        self.watermark_types = watermark_types or ['text', 'logo', 'transparent']
        
        self.video_files = self._find_video_files()
        self.frame_data = self._prepare_frame_data()

    def _find_video_files(self):
        video_files = []
        for root, _, files in os.walk(self.video_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(root, file))
        return video_files

    def _prepare_frame_data(self):
        frame_data = []
        for video_path in self.video_files:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            num_frames = min(self.max_frames_per_video, total_frames)
            frame_indices = random.sample(range(total_frames), num_frames) if total_frames > num_frames else range(total_frames)
            
            for idx in frame_indices:
                frame_data.append((video_path, idx))
        
        return frame_data

    def _add_text_watermark(self, img):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        text = random.choice(['SAMPLE', 'WATERMARK', 'COPYRIGHT', 'DEMO'])
        font_size = random.randint(20, 60)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        text_width = draw.textlength(text, font=font) if hasattr(draw, 'textlength') else 200
        x = random.randint(0, img.shape[1] - int(text_width))
        y = random.randint(0, img.shape[0] - font_size)
        
        opacity = random.randint(100, 200)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), opacity)
        
        draw.text((x, y), text, font=font, fill=color)
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _add_logo_watermark(self, img):
        h, w = img.shape[:2]
        logo_size = random.randint(30, 80)
        x = random.randint(0, w - logo_size)
        y = random.randint(0, h - logo_size)
        
        logo_img = np.zeros((logo_size, logo_size, 3), dtype=np.uint8)
        cv2.rectangle(logo_img, (0, 0), (logo_size, logo_size), 
                      (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)
        
        alpha = random.uniform(0.3, 0.7)
        roi = img[y:y+logo_size, x:x+logo_size]
        img[y:y+logo_size, x:x+logo_size] = cv2.addWeighted(roi, 1 - alpha, logo_img, alpha, 0)
        
        return img

    def _add_transparent_watermark(self, img):
        h, w = img.shape[:2]
        overlay = img.copy()
        
        num_shapes = random.randint(3, 10)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle', 'line'])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            alpha = random.uniform(0.2, 0.5)
            
            if shape_type == 'circle':
                center = (random.randint(0, w), random.randint(0, h))
                radius = random.randint(10, 50)
                cv2.circle(overlay, center, radius, color, -1)
            elif shape_type == 'rectangle':
                pt1 = (random.randint(0, w-50), random.randint(0, h-50))
                pt2 = (pt1[0] + random.randint(20, 100), pt1[1] + random.randint(20, 100))
                cv2.rectangle(overlay, pt1, pt2, color, -1)
            else:
                pt1 = (random.randint(0, w), random.randint(0, h))
                pt2 = (random.randint(0, w), random.randint(0, h))
                thickness = random.randint(2, 10)
                cv2.line(overlay, pt1, pt2, color, thickness)
        
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def _add_watermark(self, img):
        watermark_type = random.choice(self.watermark_types)
        
        if watermark_type == 'text':
            return self._add_text_watermark(img)
        elif watermark_type == 'logo':
            return self._add_logo_watermark(img)
        else:
            return self._add_transparent_watermark(img)

    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, idx):
        video_path, frame_idx = self.frame_data[idx]
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            frame = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        frame = cv2.resize(frame, self.img_size)
        clean_frame = frame.copy()
        
        watermarked_frame = self._add_watermark(frame)
        
        clean_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
        watermarked_frame = cv2.cvtColor(watermarked_frame, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            clean_frame = self.transform(clean_frame)
            watermarked_frame = self.transform(watermarked_frame)
        
        return watermarked_frame, clean_frame
