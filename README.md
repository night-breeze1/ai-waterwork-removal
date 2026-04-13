# AI 视频水印去除模型

## 项目概述

本项目提供了一个AI模型，用于自动识别并去除视频中的水印。项目基于深度学习技术，使用PyTorch框架实现。

## 功能特性

- 自动检测视频中的水印
- 自动去除识别到的水印
- 提供API接口，方便集成到其他项目
- 支持模型训练和微调
- 支持视频处理流水线

## 目录结构

```
├── src/              # 源代码目录
│   ├── simple_watermark_remover.py  # 水印去除模型
│   └── video_processor.py           # 视频处理流水线
├── app.py            # API服务
├── train_finetune_watermark_remover.py  # 模型训练和微调脚本
├── requirements.txt  # 依赖项
└── README.md         # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 启动API服务

```bash
python app.py
```

服务将运行在 http://0.0.0.0:8000

API文档地址：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 2. 训练模型

```bash
python train_finetune_watermark_remover.py --mode train
```

### 3. 微调模型

```bash
python train_finetune_watermark_remover.py --mode finetune --model_path watermark_remover.pth
```

### 4. 处理视频

```python
from src.video_processor import VideoProcessor

processor = VideoProcessor(model_path='watermark_remover.pth')
processor.process_video('input_video.mp4', 'output_video.mp4')
```

## API接口

### 1. 去除水印

- **URL**: `/api/v1/watermark/remove`
- **方法**: POST
- **参数**: 上传图像文件
- **返回**: 处理后的图像

### 2. 检测水印

- **URL**: `/api/v1/watermark/detect`
- **方法**: POST
- **参数**: 上传图像文件
- **返回**: 水印检测结果

### 3. 健康检查

- **URL**: `/api/v1/health`
- **方法**: GET
- **返回**: 服务健康状态

## 技术栈

- Python 3.8+
- PyTorch
- FastAPI
- OpenCV
- PIL
- NumPy

## 注意事项

- 模型目前仅支持处理640x480分辨率的图像
- 对于复杂的水印场景，可能需要更多的训练数据来提高效果
- 处理视频时，会在当前目录创建临时文件夹，处理完成后会自动清理

## 许可证

MIT