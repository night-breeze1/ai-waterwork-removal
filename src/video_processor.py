import cv2
import os
import torch
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time

class VideoProcessor:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        if model_path:
            from src.simple_watermark_remover import SimpleWatermarkRemover
            self.model = SimpleWatermarkRemover().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
    
    def extract_frames(self, video_path, output_dir, max_frames=None):
        """提取视频帧"""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if max_frames and saved_count >= max_frames:
                break
            
            # 保存帧
            frame_path = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
            # 显示进度
            if frame_count % 100 == 0:
                print(f'提取帧: {saved_count}/{max_frames if max_frames else ""}')
        
        cap.release()
        return saved_count
    
    def process_frame(self, frame_path):
        """处理单个帧"""
        if not self.model:
            return frame_path
        
        # 读取图像
        image = Image.open(frame_path)
        image = image.resize((640, 480))
        
        # 预处理
        img_np = np.array(image)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 127.5 - 1.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 处理
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # 后处理
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = ((output + 1.0) * 127.5).astype(np.uint8)
        
        # 保存结果
        output_path = frame_path.replace('.jpg', '_processed.jpg')
        Image.fromarray(output).save(output_path)
        
        return output_path
    
    def process_frames_batch(self, frame_paths, batch_size=4):
        """批量处理帧"""
        if not self.model:
            return frame_paths
        
        processed_paths = []
        
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i+batch_size]
            batch_images = []
            
            # 加载批量图像
            for path in batch_paths:
                image = Image.open(path)
                image = image.resize((640, 480))
                img_np = np.array(image)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 127.5 - 1.0
                batch_images.append(img_tensor)
            
            # 批量处理
            batch_tensor = torch.stack(batch_images).to(self.device)
            with torch.no_grad():
                outputs = self.model(batch_tensor)
            
            # 保存结果
            for j, output in enumerate(outputs):
                output = output.permute(1, 2, 0).cpu().numpy()
                output = ((output + 1.0) * 127.5).astype(np.uint8)
                output_path = batch_paths[j].replace('.jpg', '_processed.jpg')
                Image.fromarray(output).save(output_path)
                processed_paths.append(output_path)
            
            print(f'处理批量: {i+len(batch_paths)}/{len(frame_paths)}')
        
        return processed_paths
    
    def process_frames_parallel(self, frame_paths, max_workers=4):
        """并行处理帧"""
        processed_paths = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_frame, path) for path in frame_paths]
            for i, future in enumerate(futures):
                processed_paths.append(future.result())
                if (i+1) % 10 == 0:
                    print(f'处理帧: {i+1}/{len(frame_paths)}')
        
        return processed_paths
    
    def frames_to_video(self, frame_paths, output_video_path, fps=30):
        """将帧合成为视频"""
        if not frame_paths:
            return
        
        # 读取第一帧获取尺寸
        first_frame = cv2.imread(frame_paths[0])
        height, width, _ = first_frame.shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 写入帧
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            out.write(frame)
            if (i+1) % 100 == 0:
                print(f'合成视频: {i+1}/{len(frame_paths)}')
        
        out.release()
        return output_video_path
    
    def process_video(self, video_path, output_video_path, max_frames=None, batch_size=4, use_batch=True):
        """处理整个视频"""
        start_time = time.time()
        
        # 提取帧
        temp_dir = 'temp_frames'
        frame_count = self.extract_frames(video_path, temp_dir, max_frames)
        print(f'提取帧完成，共 {frame_count} 帧')
        
        # 获取帧路径
        frame_paths = [os.path.join(temp_dir, f'frame_{i:04d}.jpg') for i in range(frame_count)]
        
        # 处理帧
        if use_batch:
            processed_paths = self.process_frames_batch(frame_paths, batch_size)
        else:
            processed_paths = self.process_frames_parallel(frame_paths)
        print('处理帧完成')
        
        # 合成视频
        self.frames_to_video(processed_paths, output_video_path)
        print('合成视频完成')
        
        # 清理临时文件
        for path in frame_paths + processed_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        
        end_time = time.time()
        print(f'总处理时间: {end_time - start_time:.2f} 秒')
        return output_video_path