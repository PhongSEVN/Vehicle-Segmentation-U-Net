import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from model.UNet import UNet
from config.training_config import IMG_SIZE, MEAN, STD, SAVE_DIR

def predict(image_path, model_path, output_dir="results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    original_img = Image.open(image_path).convert("RGB")
    w, h = original_img.size
    
    transform = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize(mean=MEAN, std=STD)
    ])
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

    pred_mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    img_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    
    color_mask = np.zeros_like(img_bgr)
    color_mask[pred_mask_resized == 1] = [0, 255, 0] # Green cho xe
    
    alpha = 0.4
    overlay = cv2.addWeighted(img_bgr, 1.0, color_mask, alpha, 0)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"res_{filename}")
    cv2.imwrite(output_path, overlay)
    print(f"Đã lưu kết quả tại: {output_path}")

if __name__ == "__main__":
    test_image = r"D:\IT\Projects\Vehicle-Segmentation-U-Net\data\test\10_jpg.rf.550d9f60245cd17da6bda9f5f4c1a03e.jpg"
    model_weight = os.path.join(SAVE_DIR, "best_model.pth")
    
    if os.path.exists(test_image):
        predict(test_image, model_weight)
    else:
        print(f"Không tìm thấy ảnh test: {test_image}")
