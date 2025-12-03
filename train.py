import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import TerrainDataset
from src.model import DeepDSPNet
from src.losses import TVLoss, WaveletLoss

# CONFIG
ROOT_DIR = "dataset/train"
CHECKPOINT_DIR = "checkpoints"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_deepdsp_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 60
LR = 0.0001

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    ds_train = TerrainDataset(ROOT_DIR, split='train')
    ds_val = TerrainDataset(ROOT_DIR, split='val')
    
    if len(ds_train) == 0:
        print("Error: Dataset empty.")
        return

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeepDSPNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    crit_L1 = nn.L1Loss()
    crit_TV = TVLoss().to(DEVICE)
    crit_Wav = WaveletLoss().to(DEVICE)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            rgb = batch['rgb'].to(DEVICE)
            edge = batch['edges'].to(DEVICE)
            depth = batch['depth'].to(DEVICE)
            target = batch['elevation'].to(DEVICE)

            pred = model(rgb, edge, depth)
            loss = crit_L1(pred, target) + (crit_TV(pred) * 0.1) + (crit_Wav(pred, target) * 0.2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                p = model(batch['rgb'].to(DEVICE), batch['edges'].to(DEVICE), batch['depth'].to(DEVICE))
                val_loss += crit_L1(p, batch['elevation'].to(DEVICE)).item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f} | Val MAE: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f">>> Saved Best Model: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()