
# AI视频去水印项目

基于深度学习的视频水印去除系统，使用PyTorch实现的GAN架构。

## 项目结构

```
video_watermark_removal/
├── configs/              # 配置文件
│   └── default.yaml      # 默认配置
├── data/                 # 数据处理模块
│   ├── __init__.py
│   └── dataset.py        # 水印数据集
├── models/               # 模型定义
│   ├── __init__.py
│   ├── unet.py           # UNet生成器
│   └── discriminator.py  # 判别器
├── utils/                # 工具函数
│   ├── __init__.py
│   ├── losses.py         # 损失函数
│   └── metrics.py        # 评估指标
├── checkpoints/          # 模型检查点
├── logs/                 # 训练日志
├── output/               # 输出结果
├── train.py              # 训练脚本
├── inference.py          # 推理脚本
├── requirements.txt      # 依赖包
└── README.md             # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

1. 准备无水印的视频数据集
2. 将训练视频放入 `data/train/` 目录
3. 将验证视频放入 `data/val/` 目录

支持的视频格式：MP4, AVI, MOV, MKV

## 训练模型

```bash
cd video_watermark_removal
python train.py --config configs/default.yaml
```

### 恢复训练

```bash
python train.py --config configs/default.yaml --resume checkpoints/latest.pth
```

### 训练配置

编辑 `configs/default.yaml` 来调整训练参数：

- `batch_size`: 批次大小
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `img_size`: 输入图像尺寸
- 损失函数权重等

## 推理使用

### 处理单个视频

```bash
python inference.py \
    --input path/to/video.mp4 \
    --output ./output \
    --checkpoint checkpoints/latest.pth
```

### 批量处理视频

```bash
python inference.py \
    --input path/to/videos_directory \
    --output ./output \
    --checkpoint checkpoints/latest.pth \
    --batch_size 8
```

## 模型架构

- **生成器**: UNet架构，包含编码器和解码器，带跳跃连接
- **判别器**: PatchGAN架构，用于对抗训练
- **损失函数**:
  - L1损失：像素级重建
  - 感知损失：VGG特征匹配
  - 对抗损失：GAN训练
  - TV损失：平滑性约束

## 评估指标

- **PSNR**: 峰值信噪比
- **SSIM**: 结构相似性指数

## 可视化训练

使用TensorBoard查看训练过程：

```bash
tensorboard --logdir logs
```

## 提示词开发

项目根目录下的 `prompt_for_ai_watermark_removal_model.txt` 包含了详细的AI开发提示词，可以用于指导AI开发更高级的视频去水印模型。

## 注意事项

1. 训练需要大量无水印视频数据
2. 建议使用GPU加速训练
3. 自动生成的水印类型包括：文字、Logo、透明图形
4. 可以根据需要扩展自定义水印类型

## 许可证

MIT License
