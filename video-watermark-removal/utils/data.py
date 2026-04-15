import os
import cv2
import numpy as np
import random
from torch.utils.data import Dataset

# 文件路径安全性验证
def is_safe_filename(filename):
    """
    验证文件名是否安全，防止路径遍历攻击
    
    Args:
        filename: 文件名
        
    Returns:
        bool: 是否安全
    """
    return (
        filename 
        and not filename.startswith('.')
        and '..' not in filename
        and os.path.basename(filename) == filename
    )

class VideoWatermarkDataset(Dataset):
    """视频水印去除数据集"""
    def __init__(self, video_dir, watermark_dir, frame_count=16, frame_size=(256, 256), transform=None):
        """
        初始化数据集

        Args:
            video_dir: 无水印视频目录
            watermark_dir: 水印图像目录
            frame_count: 每段视频的帧数
            frame_size: 帧大小
            transform: 数据增强变换
        """
        # 输入验证
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"视频目录不存在: {video_dir}")
        if not os.path.exists(watermark_dir):
            raise FileNotFoundError(f"水印目录不存在: {watermark_dir}")
        
        self.video_dir = video_dir
        self.watermark_dir = watermark_dir
        self.frame_count = frame_count
        self.frame_size = frame_size
        self.transform = transform
        
        # 获取视频文件列表
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4') and is_safe_filename(f)]
        # 获取水印文件列表
        self.watermark_files = [f for f in os.listdir(watermark_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and is_safe_filename(f)]
        
        # 检查文件列表是否为空
        if not self.video_files:
            raise ValueError(f"视频目录中没有找到 .mp4 文件: {video_dir}")
        if not self.watermark_files:
            raise ValueError(f"水印目录中没有找到图像文件: {watermark_dir}")
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        # 加载视频
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {video_path}")
            
            # 随机选择起始帧
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_frame = random.randint(0, max(0, total_frames - self.frame_count))
            
            # 提取帧
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(self.frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                # 调整大小
                frame = cv2.resize(frame, self.frame_size)
                # 转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            if cap is not None:
                cap.release()
        
        # 确保有足够的帧
        while len(frames) < self.frame_count:
            if not frames:
                # 创建空白帧
                blank_frame = np.zeros((*self.frame_size, 3), dtype=np.float32)
                frames.append(blank_frame)
            else:
                frames.append(frames[-1])  # 重复最后一帧
        
        # 转换为numpy数组
        frames = np.array(frames, dtype=np.float32) / 255.0
        
        # 合成水印
        watermarked_frames = self.add_watermark(frames)
        
        # 应用变换
        if self.transform:
            watermarked_frames, frames = self.transform(watermarked_frames, frames)
        
        # 转换为tensor
        watermarked_frames = torch.from_numpy(watermarked_frames).permute(0, 3, 1, 2)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        return watermarked_frames, frames
    
    def add_watermark(self, frames):
        """
        为视频帧添加水印
        
        Args:
            frames: 原始视频帧
        
        Returns:
            带有水印的视频帧
        """
        watermarked_frames = frames.copy()
        
        # 随机选择水印
        watermark_path = os.path.join(self.watermark_dir, random.choice(self.watermark_files))
        watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
        
        # 调整水印大小
        h, w = self.frame_size
        max_watermark_size = min(150, w - 10, h - 10)
        watermark_size = random.randint(50, max_watermark_size)
        watermark = cv2.resize(watermark, (watermark_size, watermark_size))
        
        # 随机选择水印位置
        x = random.randint(0, max(0, w - watermark_size))
        y = random.randint(0, max(0, h - watermark_size))
        
        # 随机选择水印透明度
        alpha = random.uniform(0.1, 0.5)
        
        # 为每一帧添加水印
        for i in range(frames.shape[0]):
            frame = watermarked_frames[i]
            
            # 如果水印有alpha通道
            if watermark.shape[-1] == 4:
                # 分离alpha通道
                watermark_rgb = watermark[:, :, :3] / 255.0
                watermark_alpha = watermark[:, :, 3] / 255.0 * alpha
            else:
                # 没有alpha通道，使用整个水印
                watermark_rgb = watermark / 255.0
                watermark_alpha = np.ones((watermark_size, watermark_size)) * alpha
            
            # 混合水印
            for c in range(3):
                frame[y:y+watermark_size, x:x+watermark_size, c] = (
                    frame[y:y+watermark_size, x:x+watermark_size, c] * (1 - watermark_alpha) +
                    watermark_rgb[:, :, c] * watermark_alpha
                )
            
            watermarked_frames[i] = frame
        
        return watermarked_frames

class DataTransform:
    """数据增强变换"""
    def __init__(self, crop_size=(256, 256), flip_prob=0.5, rotate_prob=0.5, max_rotate=10):
        """
        初始化数据变换
        
        Args:
            crop_size: 裁剪大小
            flip_prob: 水平翻转概率
            rotate_prob: 旋转概率
            max_rotate: 最大旋转角度
        """
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_rotate = max_rotate
    
    def __call__(self, watermarked_frames, frames):
        """
        应用数据变换
        
        Args:
            watermarked_frames: 带有水印的帧
            frames: 原始帧
        
        Returns:
            变换后的帧
        """
        # 随机裁剪
        h, w = watermarked_frames.shape[1:3]
        crop_h, crop_w = self.crop_size
        if h > crop_h and w > crop_w:
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            watermarked_frames = watermarked_frames[:, top:top+crop_h, left:left+crop_w, :]
            frames = frames[:, top:top+crop_h, left:left+crop_w, :]
        
        # 随机水平翻转
        if random.random() < self.flip_prob:
            watermarked_frames = np.flip(watermarked_frames, axis=2)
            frames = np.flip(frames, axis=2)
        
        # 随机旋转
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.max_rotate, self.max_rotate)
            h, w = watermarked_frames.shape[1:3]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            for i in range(watermarked_frames.shape[0]):
                watermarked_frames[i] = cv2.warpAffine(watermarked_frames[i], M, (w, h))
                frames[i] = cv2.warpAffine(frames[i], M, (w, h))
        
        # 随机亮度调整
        brightness_factor = random.uniform(0.8, 1.2)
        watermarked_frames = np.clip(watermarked_frames * brightness_factor, 0, 1)
        frames = np.clip(frames * brightness_factor, 0, 1)
        
        # 随机对比度调整
        contrast_factor = random.uniform(0.8, 1.2)
        watermarked_frames = np.clip((watermarked_frames - 0.5) * contrast_factor + 0.5, 0, 1)
        frames = np.clip((frames - 0.5) * contrast_factor + 0.5, 0, 1)
        
        return watermarked_frames, frames

def create_dataloader(video_dir, watermark_dir, batch_size=8, frame_count=16, frame_size=(256, 256), shuffle=True):
    """
    创建数据加载器
    
    Args:
        video_dir: 无水印视频目录
        watermark_dir: 水印图像目录
        batch_size: 批次大小
        frame_count: 每段视频的帧数
        frame_size: 帧大小
        shuffle: 是否打乱数据
    
    Returns:
        数据加载器
    """
    transform = DataTransform()
    dataset = VideoWatermarkDataset(video_dir, watermark_dir, frame_count, frame_size, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

if __name__ == "__main__":
    # 测试数据加载器
    video_dir = "data/raw/videos"
    watermark_dir = "data/raw/watermarks"
    
    # 创建目录（如果不存在）
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(watermark_dir, exist_ok=True)
    
    # 创建测试视频
    # 这里只是示例，实际使用时需要准备真实的视频和水印
    print("创建测试数据...")
    print("请在 data/raw/videos 目录中放入无水印视频")
    print("请在 data/raw/watermarks 目录中放入水印图像")
