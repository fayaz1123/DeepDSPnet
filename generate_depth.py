import torch
import cv2
import os
import numpy as np
from PIL import Image

INPUT_DIR = "dataset/train/A"
OUTPUT_DIR = "dataset/train/C"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_depth():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading MiDaS to {DEVICE}...")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.to(DEVICE)
    model.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.tif'))])
    print(f"Processing {len(files)} images...")

    with torch.no_grad():
        for i, filename in enumerate(files):
            img = cv2.imread(os.path.join(INPUT_DIR, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            input_batch = midas_transforms(img).to(DEVICE)
            prediction = model(input_batch)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()

            depth_map = prediction.cpu().numpy()
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            save_path = os.path.join(OUTPUT_DIR, filename)
            Image.fromarray(depth_uint8).save(save_path)
            
            if i % 50 == 0: print(f"Processed {i}/{len(files)}")

if __name__ == "__main__":
    generate_depth()