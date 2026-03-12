import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from pycocotools.coco import COCO

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.training_config import DATA_ROOT

class Vehicle(Dataset):
    def __init__(self, root=DATA_ROOT, mode="train", transform=None):
        self.root = os.path.join(root, mode)
        self.transform = transform
        
        self.ann_file = os.path.join(self.root, "_annotations.coco.json")
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Không tìm thấy: {self.ann_file}")
            
        self.coco = COCO(self.ann_file)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        
        # Tạo mask từ annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Tạo mask trống (màu đen)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for ann in anns:
            m = self.coco.annToMask(ann)
            mask = np.maximum(mask, m)
        
        mask = Image.fromarray(mask * 255).convert("L")
        
        if self.transform:
            image = self.transform(image)
            if hasattr(self.transform, "transforms"):
                for t in self.transform.transforms:
                    if isinstance(t, Resize):
                        new_size = t.size[::-1] if isinstance(t.size, tuple) else (t.size, t.size)
                        mask = mask.resize(new_size, Image.NEAREST)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
        return image, mask

if __name__ == "__main__":
    dataset = Vehicle(mode="train")
    print("Tổng số ảnh:", len(dataset))

    img, mask = dataset[0]

    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy()

    img = (img * 255).astype(np.uint8)

    mask = mask.squeeze().numpy()
    mask = (mask * 255).astype(np.uint8)

    cv2.imshow("Image", img)
    cv2.imshow("Mask", mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()