import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from model.UNet import UNet
from dataset.dataset import Vehicle
from config.training_config import DATA_ROOT, IMG_SIZE, MEAN, STD, SAVE_DIR, BATCH_SIZE

def evaluate_metrics(model, dataloader, device):
    model.eval()
    
    total_dice = 0
    total_iou = 0
    total_acc = 0
    total_precision = 0
    total_recall = 0
    eps = 1e-6

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Dự đoán
            output = model(imgs)
            preds = (torch.sigmoid(output) > 0.5).float()
            
            # TP, FP, FN
            tp = (preds * masks).sum(dim=[2, 3])
            fp = (preds * (1 - masks)).sum(dim=[2, 3])
            fn = ((1 - preds) * masks).sum(dim=[2, 3])
            
            dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            total_dice += dice.mean().item()
            
            iou = (tp + eps) / (tp + fp + fn + eps)
            total_iou += iou.mean().item()
            
            precision = (tp + eps) / (tp + fp + eps)
            total_precision += precision.mean().item()
            
            recall = (tp + eps) / (tp + fn + eps)
            total_recall += recall.mean().item()
            
            acc = (preds == masks).float().mean()
            total_acc += acc.item()

    num_batches = len(dataloader)
    
    print("\n" + "="*30)
    print("      EVALUATION RESULTS")
    print("="*30)
    print(f"Dice Score (F1): {total_dice/num_batches:.4f}")
    print(f"Mean IoU:       {total_iou/num_batches:.4f}")
    print(f"Pixel Accuracy: {total_acc/num_batches:.4f}")
    print(f"Precision:      {total_precision/num_batches:.4f}")
    print(f"Recall:         {total_recall/num_batches:.4f}")
    print("="*30)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_tf = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])
    
    mode = "test" if os.path.exists(os.path.join(DATA_ROOT, "test")) else "valid"
    dataset = Vehicle(mode=mode, transform=test_tf)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    model_path = os.path.join(SAVE_DIR, "best_model.pth")
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        evaluate_metrics(model, dataloader, device)
    else:
        print(f"Không tìm thấy file model tại: {model_path}")
