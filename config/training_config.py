DATA_ROOT = r"D:\IT\Projects\Vehicle-Segmentation-U-Net\data"

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Augmentation
MIXUP = 0.4
CUTMIX = 0.4

# Normalization & Size
IMG_SIZE = 320
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Paths
SAVE_DIR = "./runs"