import os
import torch

# =====================
# Unified Configuration
# =====================

# Training dataset directories
TRAIN_IR_DIR = r'D:\com\data\MSRS\train\ir'
TRAIN_VI_DIR = r'D:\com\data\MSRS\train\vi'
TRAIN_DEPTH_DIR = r'D:\com\data\MSRS\train\depth\MSRS'  # Teacher depth directory
TRAIN_UPDATED_DEPTH_DIR = r'D:\com\data\MSRS\train\depth_updated\MSRS'  # Updated teacher depth directory, original depth is kept intact

# Algorithm 1 schedule. Trainer splits the unified MSRS training set into
# Dtrain1 and Dtrain2 deterministically.
PRETRAIN_EPOCHS = 70
NUM_SYNERGY_CYCLES = 7
FUSION_EPOCHS_PER_CYCLE = 5
JOINT_EPOCHS_PER_CYCLE = 5

# Teacher depths are generated after Stage 1 and refreshed after every cycle.
USE_DPT_FOR_DEPTH_UPDATE = True  # Use DepthAnything V2 to update depth maps
DPT_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"  # DepthAnything V2 model name for depth generation (Small version)
DPT_LOCAL_MODEL_DIR = r'./pretrained/depth-anything-v2-small-hf'  # Local Depth Anything model directory
DPT_DOWNLOAD_ENDPOINT = "https://hf-mirror.com"  # HuggingFace mirror endpoint for downloading model files

# Testing dataset directories
TEST_IR_DIR = r'D:\com\data\depth\ir'
TEST_VI_DIR = r'D:\com\data\depth\vi'

# Output / checkpoints
OUTPUT_DIR = r'C:\Users\PC\Desktop\faxiu(PR)\depth-net\100'
MODEL_DIR = r'C:\Users\PC\Desktop\faxiu(PR)\depth-net\100'  # Model save directory (lowercase)
MODEL_PATH = os.path.join(MODEL_DIR, 'checkpoint_epoch_30.pth')
LOSS_DIR = r'./loss/'  # Loss curve save directory
RESUME_CHECKPOINT = r''  # Set to None to start from scratch

# =====================
# Network Architecture Hyperparameters
# =====================

# Base channel number (controls network width)
BASE_CH = 16  # Base channel number, channels will be: 16, 32, 64, 128, 256

# Depth branch settings
DEPTH_BRANCH_IN_CH = 1  # Input channels for depth branch (1 for grayscale)
DEPTH_BRANCH_REGRESS_DEPTH = True  # Whether to regress depth prediction
USE_DEPTH = True  # Enable depth branch in network

# Fusion module settings (PaperHDGFFM)
FUSION_HEADS = 4  # Number of attention heads in fusion modules
FUSION_TAU = 1.0  # Temperature parameter for attention (controls attention sharpness)
FUSION_WITH_NORM = True  # Use GroupNorm in fusion modules

# VAR-style attention redistribution settings
VAR_ENABLED = True
VAR_ENABLE_VIS_DEPTH = False    # Apply VAR in VIS↔Depth fusion blocks
VAR_ENABLE_VIS_IR = True         # Apply calibrated VAR in bottleneck VIS↔IR fusion
VAR_SINK_DIMS = []               # Filled from ir5 after Stage 1
SINK_TOPK = 8
VAR_TAU = 20.0
VAR_P = 0.5

# =====================
# Training Hyperparameters
# =====================

BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
N_EPOCHS = PRETRAIN_EPOCHS + NUM_SYNERGY_CYCLES * (FUSION_EPOCHS_PER_CYCLE + JOINT_EPOCHS_PER_CYCLE)
LEARNING_RATE = 1e-4

# Optimizer settings (AdamW - more efficient than Adam)
ADAM_BETA1 = 0.9  # Beta1 parameter for AdamW optimizer
ADAM_BETA2 = 0.999  # Beta2 parameter for AdamW optimizer
ADAM_WEIGHT_DECAY = 1e-4  # Weight decay for AdamW (L2 regularization)

# Learning rate scheduler settings (CosineAnnealingLR)
SCHEDULER_STEP_SIZE = 10  # Legacy StepLR setting (unused)
SCHEDULER_GAMMA = 0.5  # Legacy StepLR setting (unused)
SCHEDULER_START_EPOCH = PRETRAIN_EPOCHS + 1
SCHEDULER_ETA_MIN = 1e-6  # Minimum learning rate for cosine annealing

# Legacy parameters (unused but kept for compatibility)
ALPHA = 0.25  # for SimMaxLoss/SSIM weight style
LAMBDA_L1 = 150  # legacy placeholder, if needed elsewhere

# =====================
# Loss Function Parameters
# =====================

# L_GradE (Gradient Enhancement Loss) parameters (USM-based)
# Used in: E(x) = x + λ_gain * (x - G_σ(x))
L_GRADE_LAMB = 1.5      # λ_gain: gain factor for high-frequency details
L_GRADE_SIGMA1 = 1.0    # σ: Gaussian blur std for G_σ(x)
L_GRADE_SIGMA2 = 0.0    # (unused in current USM implementation, kept for compatibility)

# Loss weights (L_total = α * L_gradE + β * L_pixel + λ * L_depth)
LOSS_WEIGHT_GRADE = 1.5   # α: weight for L_gradE
LOSS_WEIGHT_PIXEL = 1.0   # β: weight for L_pixel
LOSS_WEIGHT_DEPTH = 5     # λ: weight for L_depth (set to 0 to disable)

# Depth loss settings
DEPTH_SMOOTHNESS_SCALE = 0.1  # Scale factor for self-supervised depth smoothness loss

# Data loader / reproducibility
NUM_WORKERS = 4  # Increased from 0 to 4 for parallel data loading (adjust based on CPU cores)
SEED = 123

# Gradient accumulation settings
GRADIENT_ACCUMULATION_STEPS = 4   # Accumulate gradients over N batches (effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)

# Device policy
AUTO_USE_GPU = True    # try CUDA if available
FORCE_CPU = False      # override to force CPU
FORCE_CUDA = False     # override to force CUDA


def get_device() -> torch.device:
    """Return torch device based on policy above."""
    if FORCE_CPU:
        return torch.device('cpu')
    if FORCE_CUDA:
        if not torch.cuda.is_available():
            raise RuntimeError('FORCE_CUDA=True but CUDA is not available')
        return torch.device('cuda')
    if AUTO_USE_GPU and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOSS_DIR, exist_ok=True)


def get_dataset_paths(epoch: int):
    """
    Get dataset paths - unified dataset (no multi-dataset switching).
    
    Args:
        epoch: Current epoch number (1-indexed, not used but kept for compatibility)
    
    Returns:
        tuple: (ir_dir, vi_dir, depth_dir)
    """
    return TRAIN_IR_DIR, TRAIN_VI_DIR, TRAIN_DEPTH_DIR
