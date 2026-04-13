import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import argparse
import cv2

from src.simple_watermark_remover import SimpleWatermarkRemover

class WatermarkRemovalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.files = os.listdir(os.path.join(data_dir, 'frames'))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.data_dir, 'frames', img_name)
        image = Image.open(img_path).resize((640, 480))
        
        if self.transform:
            image = self.transform(image)
        
        # 模拟生成无水印图像（实际应用中应该使用真实的无水印图像）
        img_np = np.array(image)
        # 简单地降低水印区域的强度
        img_np = img_np * 0.8
        img_np = img_np.astype(np.uint8)
        target = Image.fromarray(img_np)
        
        if self.transform:
            target = self.transform(target)
        
        return image, target

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def finetune_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Finetune Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def generate_sample_data(data_dir, num_samples=50):
    os.makedirs(os.path.join(data_dir, 'frames'), exist_ok=True)
    for i in range(num_samples):
        # 生成随机图像
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # 添加水印
        cv2.putText(img, f'Watermark {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        img_path = os.path.join(data_dir, 'frames', f'watermark_{i}.jpg')
        Image.fromarray(img).save(img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'finetune'], help='Mode: train or finetune')
    parser.add_argument('--data_dir', type=str, default='watermark_removal_dataset', help='Directory with training data')
    parser.add_argument('--model_path', type=str, default='watermark_remover.pth', help='Path to pre-trained model for finetuning')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--finetune_epochs', type=int, default=5, help='Number of finetuning epochs')
    parser.add_argument('--save_path', type=str, default='watermark_remover.pth', help='Path to save the trained model')
    parser.add_argument('--finetune_save_path', type=str, default='finetuned_watermark_remover.pth', help='Path to save the finetuned model')
    args = parser.parse_args()
    
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成样本数据（如果数据集为空）
    if not os.path.exists(args.data_dir) or len(os.listdir(os.path.join(args.data_dir, 'frames'))) == 0:
        print('Generating sample data...')
        generate_sample_data(args.data_dir)
    
    # 数据变换
    transform = lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 127.5 - 1.0
    
    # 数据集和数据加载器
    dataset = WatermarkRemovalDataset(args.data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 模型
    if args.mode == 'train':
        model = SimpleWatermarkRemover().to(device)
    else:  # finetune
        model = SimpleWatermarkRemover().to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练或微调
    if args.mode == 'train':
        print('Training model...')
        train_model(model, train_loader, criterion, optimizer, num_epochs=args.epochs)
        torch.save(model.state_dict(), args.save_path)
        print(f'Model saved to {args.save_path}')
    else:
        print('Finetuning model...')
        finetune_model(model, train_loader, criterion, optimizer, num_epochs=args.finetune_epochs)
        torch.save(model.state_dict(), args.finetune_save_path)
        print(f'Finetuned model saved to {args.finetune_save_path}')