import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """注意力机制模块"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 生成注意力掩码
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        # 应用注意力掩码
        return x * attention

class DownBlock(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # 保存跳跃连接
        skip = x
        # 下采样
        x = self.pool(x)
        return x, skip

class UpBlock(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = AttentionBlock(out_channels)
    
    def forward(self, x, skip):
        # 上采样
        x = self.up(x)
        # 拼接跳跃连接
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # 应用注意力机制
        x = self.attention(x)
        return x

class TimeBlock(nn.Module):
    """时间序列处理模块"""
    def __init__(self, in_channels, out_channels):
        super(TimeBlock, self).__init__()
        # 3D卷积来捕获时间信息
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 输入形状: (batch, frames, channels, height, width)
        # 转换为3D卷积输入形状: (batch, channels, frames, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        # 转换回原始形状
        x = x.permute(0, 2, 1, 3, 4)
        return x

class VideoUNet(nn.Module):
    """视频水印去除模型"""
    def __init__(self, in_channels=3, out_channels=3):
        super(VideoUNet, self).__init__()
        
        # 时间处理模块
        self.time_block = TimeBlock(in_channels, in_channels)
        
        # 编码器
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        
        # 瓶颈
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        
        # 输出层
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # 输入形状: (batch, frames, channels, height, width)
        batch_size, frames, channels, height, width = x.shape
        
        # 处理时间信息
        x = self.time_block(x)
        
        # 处理每一帧
        outputs = []
        for i in range(frames):
            frame = x[:, i, :, :, :]
            
            # 编码器
            d1, skip1 = self.down1(frame)
            d2, skip2 = self.down2(d1)
            d3, skip3 = self.down3(d2)
            d4, skip4 = self.down4(d3)
            
            # 瓶颈
            bn = self.bottleneck(d4)
            
            # 解码器
            u1 = self.up1(bn, skip4)
            u2 = self.up2(u1, skip3)
            u3 = self.up3(u2, skip2)
            u4 = self.up4(u3, skip1)
            
            # 输出
            out = self.output(u4)
            outputs.append(out)
        
        # 合并帧
        outputs = torch.stack(outputs, dim=1)
        return outputs

if __name__ == "__main__":
    # 测试模型
    model = VideoUNet()
    # 输入形状: (batch, frames, channels, height, width)
    input = torch.randn(1, 16, 3, 256, 256)
    output = model(input)
    print(f"输入形状: {input.shape}")
    print(f"输出形状: {output.shape}")
