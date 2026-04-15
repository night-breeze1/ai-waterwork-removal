import pytest
import torch
import numpy as np
from utils.metrics import calculate_psnr, calculate_ssim

def test_calculate_psnr_identical():
    pred = torch.randn(1, 1, 3, 64, 64)
    target = pred.clone()
    psnr = calculate_psnr(pred, target)
    assert psnr == float('inf')

def test_calculate_psnr_different():
    pred = torch.randn(1, 1, 3, 64, 64)
    target = torch.randn(1, 1, 3, 64, 64)
    psnr = calculate_psnr(pred, target)
    assert isinstance(psnr, float)
    assert psnr < 100

def test_calculate_ssim():
    pred = torch.rand(1, 1, 3, 64, 64)
    target = torch.rand(1, 1, 3, 64, 64)
    ssim_val = calculate_ssim(pred, target)
    assert isinstance(ssim_val, float)
    assert 0 <= ssim_val <= 1

def test_calculate_ssim_identical():
    pred = torch.rand(1, 1, 3, 64, 64)
    target = pred.clone()
    ssim_val = calculate_ssim(pred, target)
    assert ssim_val > 0.99
