import os
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from model.UNet import UNet
from config.training_config import DATA_ROOT, IMG_SIZE, MEAN, STD, SAVE_DIR

def run_test_batch(test_dir, model_path, output_dir="results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model tại: {model_path}")
        return
        
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    transform = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize(mean=MEAN, std=STD)
    ])

    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    if not os.path.exists(test_dir):
        print(f"Thư mục test không tồn tại: {test_dir}")
        return
        
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(image_exts)]
    
    if not image_files:
        print(f"Không tìm thấy ảnh nào trong thư mục: {test_dir}")
        return

    print(f"Bắt đầu dự đoán {len(image_files)} ảnh...")

    with torch.no_grad():
        for filename in tqdm(image_files, desc="Batch Predicting"):
            img_path = os.path.join(test_dir, filename)
            
            try:
                original_img = Image.open(img_path).convert("RGB")
                w, h = original_img.size
                input_tensor = transform(original_img).unsqueeze(0).to(device)

                # Inference
                output = model(input_tensor)
                pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8)

                pred_mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                img_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
                color_mask = np.zeros_like(img_bgr)
                color_mask[pred_mask_resized == 1] = [0, 255, 0] # Màu xanh lá
                
                overlay = cv2.addWeighted(img_bgr, 0.7, color_mask, 0.3, 0)
                cv2.imwrite(os.path.join(output_dir, f"res_{filename}"), overlay)
            except Exception as e:
                print(f"⚠Lỗi xử lý {filename}: {e}")

    print(f"Kết quả lưu tại: {output_dir}")

if __name__ == "__main__":
    path_test = os.path.join(DATA_ROOT, "test")
    path_model = os.path.join(SAVE_DIR, "best_model.pth")
    run_test_batch(path_test, path_model)
