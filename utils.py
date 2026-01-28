import os
import torch

# =====================
# Unified Configuration
# =====================

# Training dataset directories
TRAIN_IR_DIR = r'D:\com\data\MSRS\ir'
TRAIN_VI_DIR = r'D:\com\data\MSRS\vi'
TRAIN_DEPTH_DIR = r'D:\com\data\MSRS\train\depth\MSRS'  # Teacher depth directory

# Multi-dataset training (cyclic dataset switching) - COMMENTED OUT
TRAIN1_BASE_DIR = r'D:\com\data\MSRS\train\meta\train1'  # Used in epochs 1-3, 5-7, 9-11, ...
TRAIN2_BASE_DIR = r'D:\com\data\MSRS\train\meta\train2'  # Used in epochs 4, 8, 12, ...
USE_MULTI_DATASET = True  # Disable multi-dataset training, use unified dataset

# Cycle configuration: (TRAIN1_EPOCHS_PER_CYCLE train1 + TRAIN2_EPOCHS_PER_CYCLE train2) - COMMENTED OUT
TRAIN1_EPOCHS_PER_CYCLE = 3  # Number of epochs using train1 in each cycle
TRAIN2_EPOCHS_PER_CYCLE = 2  # Number of epochs using train2 in each cycle
# CYCLE_LENGTH = TRAIN1_EPOCHS_PER_CYCLE + TRAIN2_EPOCHS_PER_CYCLE  # Total epochs per cycle (automatically calculated)

# Depth map update settings
DEPTH_UPDATE_START_EPOCH = 50  # Start updating depth maps from this epoch
DEPTH_UPDATE_INTERVAL = 10  # Update depth maps every N epochs (e.g., 120, 130, 140, ...)
USE_DPT_FOR_DEPTH_UPDATE = True  # Use DepthAnything V2 to update depth maps
DPT_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"  # DepthAnything V2 model name for depth generation (Small version)

# Testing dataset directories
TEST_IR_DIR = r'D:\com\data\MSRS\ir'
TEST_VI_DIR = r'D:\com\data\MSRS\vi'

# Output / checkpoints
OUTPUT_DIR = r'./Output/'
MODEL_DIR = r'./model/'  # Model save directory (lowercase)
MODEL_PATH = os.path.join(MODEL_DIR, 'checkpoint_epoch_119.pth')
LOSS_DIR = r'./loss/'  # Loss curve save directory

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
VAR_ENABLE_VIS_IR = True        # Apply VAR in bottleneck VIS↔IR fusion
VAR_SINK_DIMS = [33, 136, 169, 192, 142, 144, 200, 134]  # indices derived from ir5
VAR_TAU = 20.0
VAR_P = 0.5

# =====================
# Training Hyperparameters
# =====================

BATCH_SIZE = 2
TEST_BATCH_SIZE = 1
N_EPOCHS = 100
LEARNING_RATE = 1e-4

# Optimizer settings (AdamW - more efficient than Adam)
ADAM_BETA1 = 0.9  # Beta1 parameter for AdamW optimizer
ADAM_BETA2 = 0.999  # Beta2 parameter for AdamW optimizer
ADAM_WEIGHT_DECAY = 1e-4  # Weight decay for AdamW (L2 regularization)

# Learning rate scheduler settings (StepLR)
SCHEDULER_STEP_SIZE = 10  # Decay learning rate every N epochs
SCHEDULER_GAMMA = 0.5  # Multiply learning rate by gamma at each step
SCHEDULER_START_EPOCH = 50  # Start decaying from this epoch

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

# L_Mask (Mask Loss) parameters (currently not used in total loss)
L_MASK_SIGMA_LOW = 1.0  # sigma for low-frequency Gaussian blur (legacy)

# Loss weights (for L_total = α * L_gradE + β * L_pixel + λ * L_depth)
LOSS_WEIGHT_GRADE = 1.5   # α: weight for L_gradE
LOSS_WEIGHT_PIXEL = 1.0   # β: weight for L_pixel
LOSS_WEIGHT_MASK = 0.0    # γ: weight for L_mask (set to 0 -> disabled)
LOSS_WEIGHT_DEPTH = 5     # λ: weight for L_depth (set to 0 to disable)

# Depth loss settings
DEPTH_SMOOTHNESS_SCALE = 0.1  # Scale factor for self-supervised depth smoothness loss

# Data loader / reproducibility
NUM_WORKERS = 4  # Increased from 0 to 4 for parallel data loading (adjust based on CPU cores)
SEED = 123

# Gradient accumulation settings
GRADIENT_ACCUMULATION_STEPS = 1   # Accumulate gradients over N batches (effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)

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
    # Always use unified training dataset
    return TRAIN_IR_DIR, TRAIN_VI_DIR, TRAIN_DEPTH_DIR
    
    # Multi-dataset training code (COMMENTED OUT)
    # if not USE_MULTI_DATASET:
    #     # Use default paths
    #     return TRAIN_IR_DIR, TRAIN_VI_DIR, TRAIN_DEPTH_DIR
    # 
    # # Calculate position in cycle (1-indexed)
    # # epoch=1,2,3,4 -> cycle_pos=1,2,3,4 (if CYCLE_LENGTH=4)
    # cycle_pos = ((epoch - 1) % CYCLE_LENGTH) + 1
    # 
    # # Use train2 if cycle_pos > TRAIN1_EPOCHS_PER_CYCLE
    # # Otherwise use train1
    # if cycle_pos > TRAIN1_EPOCHS_PER_CYCLE:
    #     # Use train2 for epochs after TRAIN1_EPOCHS_PER_CYCLE in each cycle
    #     base_dir = TRAIN2_BASE_DIR
    #     dataset_name = 'train2'
    # else:
    #     # Use train1 for first TRAIN1_EPOCHS_PER_CYCLE epochs in each cycle
    #     base_dir = TRAIN1_BASE_DIR
    #     dataset_name = 'train1'
    # 
    # # Assume subdirectories: ir, vi, depth (or depth/MSRS)
    # ir_dir = os.path.join(base_dir, 'ir')
    # vi_dir = os.path.join(base_dir, 'vi')
    # 
    # # Try depth/MSRS first, then depth
    # depth_dir1 = os.path.join(base_dir, 'depth', 'MSRS')
    # depth_dir2 = os.path.join(base_dir, 'depth')
    # if os.path.exists(depth_dir1):
    #     depth_dir = depth_dir1
    # elif os.path.exists(depth_dir2):
    #     depth_dir = depth_dir2
    # else:
    #     depth_dir = None  # No depth directory
    # 
    # return ir_dir, vi_dir, depth_dir


