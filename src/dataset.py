import os
import random
import numpy as np
import torch
import cv2
import rasterio
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class TerrainDataset(Dataset):
    def __init__(self, root_dir, split='train', seed=42):
        self.root_dir = root_dir
        self.split = split
        self.path_A = os.path.join(root_dir, 'A') # RGB
        self.path_B = os.path.join(root_dir, 'B') # Elevation
        self.path_C = os.path.join(root_dir, 'C') # Depth
        
        if os.path.exists(self.path_A):
            all_files = sorted([
                f for f in os.listdir(self.path_A) 
                if f.lower().endswith(('.png', '.tif', '.jpg', '.jpeg')) and not f.startswith('.')
            ])
        else:
            all_files = []

        # Deterministic split
        random.seed(seed)
        random.shuffle(all_files)
        
        total_files = len(all_files)
        train_idx = int(0.6 * total_files) 
        val_idx = int(0.8 * total_files)   
        
        if split == 'train':
            self.image_filenames = all_files[:train_idx]
        elif split == 'val':
            self.image_filenames = all_files[train_idx:val_idx]
        elif split == 'test':
            self.image_filenames = all_files[val_idx:]

        self.has_depth_cache = os.path.exists(self.path_C)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_filenames[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # 1. RGB
        rgb_path = os.path.join(self.path_A, img_name)
        try:
            rgb_image = Image.open(rgb_path).convert('RGB')
            rgb_np = np.array(rgb_image)
        except Exception:
            # Fallback to next image if load fails
            return self.__getitem__((idx + 1) % len(self))

        # 2. Edges
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_pil = Image.fromarray(edges)

        # 3. Depth
        depth_pil = None
        if self.has_depth_cache:
            depth_path = os.path.join(self.path_C, base_name + ".png") # Check png
            if not os.path.exists(depth_path):
                depth_path = os.path.join(self.path_C, img_name) # Check original ext
            
            if os.path.exists(depth_path):
                try:
                    depth_pil = Image.open(depth_path).convert('L')
                except: pass
        
        # Fallback Depth (Laplacian)
        if depth_pil is None:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.absolute(laplacian)
            laplacian = (laplacian / (laplacian.max() + 1e-6) * 255.0).astype(np.uint8)
            depth_pil = Image.fromarray(laplacian)

        # 4. Elevation
        dem_path = os.path.join(self.path_B, base_name + ".tif")
        try:
            with rasterio.open(dem_path) as src:
                elev_np = src.read(1)
                elev_np[elev_np < -100] = 0.0 # Filter bad data
                elev_tensor = torch.from_numpy(elev_np).float().unsqueeze(0)
        except:
            elev_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)

        to_tensor = transforms.ToTensor()
        return {
            'rgb': to_tensor(rgb_image),
            'edges': to_tensor(edges_pil),
            'depth': to_tensor(depth_pil),
            'elevation': elev_tensor,
            'filename': base_name
        }