#!/usr/bin/env python3
"""
代码验证脚本
验证第一阶段代码完善的成果
"""

import os
import sys
import importlib

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== 视频水印去除AI项目代码验证 ===")
print()

def test_imports():
    """测试模块导入"""
    print("1. 测试模块导入...")
    modules = [
        'models.architectures.video_unet',
        'utils.data',
        'utils.metrics',
        'utils.visualization',
        'inference.infer',
        'inference.demo',
        'training.train',
        'training.evaluate'
    ]
    
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"   ✅ {module_name}")
        except Exception as e:
            print(f"   ❌ {module_name}: {e}")
    print()

def test_model_structure():
    """测试模型结构"""
    print("2. 测试模型结构...")
    try:
        from models.architectures.video_unet import VideoUNet
        model = VideoUNet()
        print(f"   ✅ 模型创建成功: {model.__class__.__name__}")
        
        # 测试模型前向传播
        import torch
        input_tensor = torch.randn(1, 2, 3, 64, 64)
        output = model(input_tensor)
        print(f"   ✅ 前向传播成功: 输入形状={input_tensor.shape}, 输出形状={output.shape}")
    except Exception as e:
        print(f"   ❌ 模型测试失败: {e}")
    print()

def test_utils():
    """测试工具函数"""
    print("3. 测试工具函数...")
    
    # 测试is_safe_filename
    try:
        from utils.data import is_safe_filename
        test_cases = ["test.mp4", ".hidden.txt", "../test.txt", "test.png"]
        for test_case in test_cases:
            result = is_safe_filename(test_case)
            print(f"   ✅ is_safe_filename('{test_case}') = {result}")
    except Exception as e:
        print(f"   ❌ 测试is_safe_filename失败: {e}")
    
    # 测试metrics
    try:
        from utils.metrics import calculate_psnr
        import torch
        pred = torch.randn(1, 1, 3, 32, 32)
        target = torch.randn(1, 1, 3, 32, 32)
        psnr = calculate_psnr(pred, target)
        print(f"   ✅ calculate_psnr 测试成功: {psnr:.2f} dB")
    except Exception as e:
        print(f"   ❌ 测试metrics失败: {e}")
    print()

def test_inference_scripts():
    """测试推理脚本"""
    print("4. 测试推理脚本...")
    
    # 测试infer.py命令行参数解析
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, 'inference/infer.py', '--help'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   ✅ infer.py 命令行参数解析成功")
        else:
            print(f"   ❌ infer.py 命令行参数解析失败: {result.stderr}")
    except Exception as e:
        print(f"   ❌ 测试infer.py失败: {e}")
    
    # 测试demo.py命令行参数解析
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, 'inference/demo.py', '--help'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   ✅ demo.py 命令行参数解析成功")
        else:
            print(f"   ❌ demo.py 命令行参数解析失败: {result.stderr}")
    except Exception as e:
        print(f"   ❌ 测试demo.py失败: {e}")
    print()

def test_training_scripts():
    """测试训练脚本"""
    print("5. 测试训练脚本...")
    
    # 测试train.py命令行参数解析
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, 'training/train.py', '--help'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   ✅ train.py 命令行参数解析成功")
        else:
            print(f"   ❌ train.py 命令行参数解析失败: {result.stderr}")
    except Exception as e:
        print(f"   ❌ 测试train.py失败: {e}")
    
    # 测试evaluate.py命令行参数解析
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, 'training/evaluate.py', '--help'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   ✅ evaluate.py 命令行参数解析成功")
        else:
            print(f"   ❌ evaluate.py 命令行参数解析失败: {result.stderr}")
    except Exception as e:
        print(f"   ❌ 测试evaluate.py失败: {e}")
    print()

if __name__ == "__main__":
    test_imports()
    test_model_structure()
    test_utils()
    test_inference_scripts()
    test_training_scripts()
    print("=== 验证完成 ===")
