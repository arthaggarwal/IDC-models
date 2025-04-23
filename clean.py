import torch
from models.yolo import DetectionModel
from torch.serialization import add_safe_globals
from pathlib import WindowsPath

# Allow safe loading of DetectionModel
add_safe_globals([DetectionModel])

# Load full checkpoint
ckpt = torch.load('yolov5/runs/train/exp9/weights/best.pt', map_location='cpu', weights_only=False)

# 🔧 Recursively convert all WindowsPaths to strings
def convert_paths(obj):
    if isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_paths(i) for i in obj)
    elif isinstance(obj, WindowsPath):
        return str(obj)
    else:
        return obj

ckpt_cleaned = convert_paths(ckpt)

# Save the cleaned checkpoint
torch.save(ckpt_cleaned, 'best_clean(2).pt')
print("✅ Cleaned checkpoint saved without WindowsPath")
