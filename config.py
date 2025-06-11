import torch

# Dataset and paths
class Config:
    DATA_ROOT = "/home/step/data/camreid/new/data/sysu_mm01/SYSU-MM01"
    MODEL_PATH = "checkpoints/reid_model.pth"

    # Model parameters
    NUM_CLASSES = 395  # SYSU-MM01 identities
    OUTPUT_DIM = 576   # MobileNetV3 feature dimension
    K_REGIONS = 3      # Sparse regions for alignment
    G_GROUPS = 4       # Grouped attention groups
    K_SHOTS = 5        # Few-shot samples per class

    # Training parameters
    BATCH_SIZE = 12
    NUM_EPOCHS = 1
    LR = 1e-3
    LAMBDA_1 = 0.5  # Weight for MMD loss
    LAMBDA_2 = 0.3  # Weight for alignment loss
    LAMBDA_3 = 1.0  # Weight for ID loss

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()