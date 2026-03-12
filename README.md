# 🚗 Vehicle Segmentation using U-Net

Dự án này triển khai mô hình **U-Net** để phân đoạn (segmentation) phương tiện giao thông, sử dụng bộ dữ liệu **Vehicle ETFSO** (COCO Format).
---

## 🛠️ Cấu trúc dự án
```text
Vehicle-Segmentation-U-Net/
├── config/
│   └── training_config.py   # Tất cả cấu hình nằm tại đây
├── dataset/
│   └── dataset.py           # Xử lý COCO JSON & Augmentations
├── model/
│   └── UNet.py              # Kiến trúc mô hình U-Net
├── data/                    # Thư mục chứa dữ liệu (train/valid)
├── runs/                    # Lưu trữ logs & weights
└── train.py                 # Script huấn luyện chính
```

---

## ⚙️ Cài đặt

1. **Clone repository**
   ```bash
   git clone https://github.com/PhongSEVN/Vehicle-Segmentation-U-Net.git
   cd Vehicle-Segmentation-U-Net
   ```
2. **Cài đặt môi trường**
   ```bash
   pip install -r requirements.txt
   ```
3. **Chuẩn bị dữ liệu**
   - Tải dataset từ [Roboflow (COCO Format)](https://universe.roboflow.com/vehi/vehicle-etfso).
   - Giải nén vào thư mục `data/` sao cho đường dẫn có dạng: `data/train/_annotations.coco.json`.

---

## 🚀 Huấn luyện (Training)

Toàn bộ thông số huấn luyện được quản lý tập trung tại `config/training_config.py`. Bạn có thể tùy chỉnh `BATCH_SIZE`, `LEARNING_RATE`, hoặc bật/tắt `MIXUP/CUTMIX` tại đó.

**Chạy huấn luyện:**
```bash
python train.py
```

**Các tính năng nổi bật:**
- **TQDM Bar:** Theo dõi tiến độ huấn luyện thời gian thực.
- **Mixed Augmentations:** Tích hợp Mixup và Cutmix để xử lý vấn đề mất cân bằng dữ liệu (đặc biệt cho lớp xe máy).
- **Scheduler:** Sử dụng CosineAnnealingLR để tối ưu hóa việc hội tụ.

---

## 📊 Giám sát & Kết quả

### 1. TensorBoard
Dự án ghi lại loss, dice và visualize kết quả trực quan. Để xem:
```bash
tensorboard --logdir=runs/logs
```

### 2. Kiểm tra Dataset
Bạn có thể chạy trực tiếp file dataset để kiểm tra xem mask có khớp với ảnh không:
```bash
python dataset/dataset.py
```

---

## 📈 Chỉ số đánh giá
Đang huấn luyện trên GPU, cập nhật sau