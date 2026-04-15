import pytest
import torch
from models.architectures.video_unet import VideoUNet, AttentionBlock, DownBlock, UpBlock, TimeBlock

def test_attention_block():
    block = AttentionBlock(64)
    x = torch.randn(2, 64, 32, 32)
    output = block(x)
    assert output.shape == x.shape

def test_down_block():
    block = DownBlock(3, 64)
    x = torch.randn(2, 3, 64, 64)
    output, skip = block(x)
    assert output.shape == (2, 64, 32, 32)
    assert skip.shape == (2, 64, 64, 64)

def test_up_block():
    block = UpBlock(128, 64)
    x = torch.randn(2, 128, 32, 32)
    skip = torch.randn(2, 64, 64, 64)
    output = block(x, skip)
    assert output.shape == (2, 64, 64, 64)

def test_time_block():
    block = TimeBlock(3, 3)
    x = torch.randn(2, 4, 3, 64, 64)
    output = block(x)
    assert output.shape == x.shape

def test_video_unet():
    model = VideoUNet()
    x = torch.randn(1, 16, 3, 256, 256)
    output = model(x)
    assert output.shape == x.shape
