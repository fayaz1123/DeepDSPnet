import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import TerrainDataset
from src.model import DeepDSPNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_deepdsp_model.pth"
OUTPUT_DIR = "results"
ROOT_DIR = "dataset/train"

def run_inference():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    test_ds = TerrainDataset(ROOT_DIR, split='test')
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    model = DeepDSPNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    saved_count = 0
    with torch.no_grad():
        for batch in test_loader:
            if saved_count >= 20: break 
            
            rgb = batch['rgb'].to(DEVICE)
            edge = batch['edges'].to(DEVICE)
            depth = batch['depth'].to(DEVICE)
            target = batch['elevation'].to(DEVICE)
            filenames = batch['filename']
            
            preds = model(rgb, edge, depth)
            
            for i in range(len(rgb)):
                img_rgb = rgb[i].permute(1, 2, 0).cpu().numpy()
                img_target = target[i].squeeze().cpu().numpy()
                img_pred = preds[i].squeeze().cpu().numpy()
                error_map = np.abs(img_target - img_pred)
                
                fig, axs = plt.subplots(1, 4, figsize=(20, 5))
                
                # Plotting
                axs[0].imshow(img_rgb)
                axs[0].set_title("Input")
                
                vmin, vmax = min(img_target.min(), img_pred.min()), max(img_target.max(), img_pred.max())
                
                im1 = axs[1].imshow(img_target, cmap='terrain', vmin=vmin, vmax=vmax)
                axs[1].set_title("Target")
                plt.colorbar(im1, ax=axs[1], fraction=0.046)
                
                im2 = axs[2].imshow(img_pred, cmap='terrain', vmin=vmin, vmax=vmax)
                axs[2].set_title("Prediction")
                plt.colorbar(im2, ax=axs[2], fraction=0.046)
                
                im3 = axs[3].imshow(error_map, cmap='hot')
                axs[3].set_title("Error Map")
                plt.colorbar(im3, ax=axs[3], fraction=0.046).set_label('Meters')
                
                for ax in axs: ax.axis('off')
                
                plt.savefig(f"{OUTPUT_DIR}/{filenames[i]}_result.png", bbox_inches='tight', dpi=100)
                plt.close()
                saved_count += 1
                print(f"Saved result for {filenames[i]}")

if __name__ == "__main__":
    run_inference()