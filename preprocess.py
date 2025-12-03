import rasterio
from rasterio.windows import Window
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

PATCH_SIZE = 512
STRIDE = 256
OUTPUT_DIR = "dataset/train"

DATA_SOURCES = [
    {"name": "norfolk", "rgb": "data/train_rgb_nir.tif", "dem": "data/train_elevation.tif"},
    {"name": "miami", "rgb": "data/miami_rgb.tif", "dem": "data/miami_dem.tif"}
]

def create_dirs():
    os.makedirs(f"{OUTPUT_DIR}/A", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/B", exist_ok=True)

def slice_data():
    create_dirs()
    total_count = 0

    for source in DATA_SOURCES:
        if not os.path.exists(source["rgb"]) or not os.path.exists(source["dem"]):
            print(f"Skipping {source['name']}: Files not found.")
            continue

        with rasterio.open(source["rgb"]) as src_rgb, rasterio.open(source["dem"]) as src_dem:
            cols = (src_rgb.width - PATCH_SIZE) // STRIDE
            rows = (src_rgb.height - PATCH_SIZE) // STRIDE
            
            with tqdm(total=cols*rows, desc=f"Slicing {source['name']}") as pbar:
                for i in range(cols):
                    for j in range(rows):
                        window = Window(i * STRIDE, j * STRIDE, PATCH_SIZE, PATCH_SIZE)
                        rgb_patch = src_rgb.read(window=window) 
                        dem_patch = src_dem.read(1, window=window)
                        
                        if np.mean(rgb_patch) == 0 or np.min(dem_patch) < -100:
                            pbar.update(1)
                            continue

                        file_id = f"{source['name']}_{i}_{j}"
                        
                        # Save RGB
                        rgb_save = np.moveaxis(rgb_patch[:3], 0, -1).astype('uint8')
                        Image.fromarray(rgb_save).save(f"{OUTPUT_DIR}/A/{file_id}.png")
                        
                        # Save DEM (Float32 TIF)
                        profile = src_dem.profile
                        profile.update({'driver': 'GTiff', 'height': PATCH_SIZE, 'width': PATCH_SIZE, 'count': 1, 'dtype': rasterio.float32})
                        with rasterio.open(f"{OUTPUT_DIR}/B/{file_id}.tif", 'w', **profile) as dst:
                            dst.write(dem_patch, 1)

                        total_count += 1
                        pbar.update(1)

    print(f"Done. {total_count} pairs created in {OUTPUT_DIR}.")

if __name__ == "__main__":
    slice_data()