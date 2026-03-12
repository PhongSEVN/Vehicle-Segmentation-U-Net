import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, RandomAffine, ColorJitter, ToTensor, Normalize, Resize
from torchvision.utils import make_grid

from model.UNet import UNet
from dataset.dataset import Vehicle
from config.training_config import (
    DATA_ROOT, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MIXUP, CUTMIX, IMG_SIZE, MEAN, STD, SAVE_DIR
)


def get_transforms(img_size):
    train_tf = Compose([
        RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ColorJitter(brightness=0.2, contrast=0.2),
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])
    val_tf = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])
    return train_tf, val_tf

# --- Augmentations ---
def apply_mixup(imgs, masks, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(imgs.size(0)).to(imgs.device)
    return lam * imgs + (1 - lam) * imgs[index, :], lam * masks + (1 - lam) * masks[index, :]

def apply_cutmix(imgs, masks, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(imgs.size(0)).to(imgs.device)
    W, H = imgs.size(2), imgs.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[index, :, bbx1:bbx2, bby1:bby2]
    masks[:, :, bbx1:bbx2, bby1:bby2] = masks[index, :, bbx1:bbx2, bby1:bby2]
    return imgs, masks

def dice_coeff(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=[2, 3])
    union = pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3])
    return ((2.0 * inter + eps) / (union + eps)).mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(SAVE_DIR) / "logs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Transforms
    train_tf = Compose([
        RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ColorJitter(brightness=0.2, contrast=0.2),
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])
    val_tf = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])

    train_ds = Vehicle(mode="train", transform=train_tf)
    val_ds = Vehicle(mode="valid", transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_dice = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_dice = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            r = np.random.rand()
            if r < MIXUP: imgs, masks = apply_mixup(imgs, masks)
            elif r < MIXUP + CUTMIX: imgs, masks = apply_cutmix(imgs, masks)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            curr_dice = dice_coeff(preds, masks).item()
            train_dice += curr_dice * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{curr_dice:.4f}")

        train_loss /= len(train_ds); train_dice /= len(train_ds)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Dice/Train", train_dice, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        model.eval()
        val_loss, val_dice = 0, 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]")
        with torch.no_grad():
            for i, (imgs, masks) in enumerate(pbar_val):
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss_v = criterion(preds, masks).item()
                dice_v = dice_coeff(preds, masks).item()
                val_loss += loss_v * imgs.size(0)
                val_dice += dice_v * imgs.size(0)
                pbar_val.set_postfix(loss=f"{loss_v:.4f}", dice=f"{dice_v:.4f}")
                
                if i == 0:
                    vis_preds = torch.sigmoid(preds) > 0.5
                    img_grid = make_grid(imgs[:8], normalize=True)
                    mask_grid = make_grid(masks[:8])
                    pred_grid = make_grid(vis_preds[:8].float())
                    writer.add_image("Val/Images", img_grid, epoch)
                    writer.add_image("Val/Masks_GT", mask_grid, epoch)
                    writer.add_image("Val/Predictions", pred_grid, epoch)

        val_loss /= len(val_ds); val_dice /= len(val_ds)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Dice/Val", val_dice, epoch)
        
        scheduler.step()
        print(f"Summary -> Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Path(SAVE_DIR) / "best_model.pth")
            print(f"Save Best Model (Dice: {best_dice:.4f})")

    writer.close()
    print(f"Huấn luyện hoàn tất! Xem logs tại {Path(SAVE_DIR)/'logs'}")

if __name__ == "__main__":
    train()
