from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import cv2
import numpy as np
import torch
from PIL import Image
import tempfile
import os

from src.simple_watermark_remover import SimpleWatermarkRemover

app = FastAPI()

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleWatermarkRemover().to(device)

# 尝试加载预训练模型
try:
    model.load_state_dict(torch.load('watermark_remover.pth', map_location=device))
    model.eval()
except:
    pass

@app.post("/api/v1/watermark/remove")
async def remove_watermark(file: UploadFile = File(...)):
    # 读取图像
    image = Image.open(file.file)
    image = image.resize((640, 480))
    
    # 预处理
    img_np = np.array(image)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 127.5 - 1.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 处理
    with torch.no_grad():
        output = model(img_tensor)
    
    # 后处理
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = ((output + 1.0) * 127.5).astype(np.uint8)
    
    # 保存结果
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
        temp_path = temp.name
        Image.fromarray(output).save(temp_path)
    
    return FileResponse(temp_path, media_type="image/jpeg")

@app.post("/api/v1/watermark/detect")
async def detect_watermark(file: UploadFile = File(...)):
    # 简单的水印检测实现
    image = Image.open(file.file)
    img_np = np.array(image)
    
    # 转换为灰度
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 100, 200)
    
    # 计算边缘密度
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    # 简单判断
    has_watermark = edge_density > 0.05
    
    return {"has_watermark": has_watermark, "edge_density": edge_density}

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)