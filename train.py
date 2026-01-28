#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Fusion Network Training Script
This script implements training pipeline for an image fusion network.
"""

import argparse
import os
import time
from typing import Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Tkinter thread issues
import matplotlib.pyplot as plt

from moudle import fusiondata
from net import Net
from loss import L_GradE, L_Int, L_Mask
import utils


def normalize_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each image in batch to [0,1] range."""
    B, C, H, W = x.shape
    x_ = x.view(B, C, -1)
    mn = x_.min(dim=2, keepdim=True)[0]
    mx = x_.max(dim=2, keepdim=True)[0]
    x_ = (x_ - mn) / (mx - mn + eps)
    return x_.view(B, C, H, W)


# ==========================
# Depth Map Generation Functions
# ==========================

def ensure_dir(p: Path):
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def minmax_norm(x, eps=1e-6):
    """Normalize array to [0,1] range."""
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min + eps)


def load_midas(model_name="DPT_Large", device="cuda"):
    """Load MiDaS model and transforms."""
    midas = torch.hub.load("intel-isl/MiDaS", model_name)
    midas.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform if model_name in ["DPT_Large", "DPT_Hybrid"] else transforms.small_transform
    return midas, transform


def predict_depth(midas, transform, img_rgb, device="cuda"):
    """Predict depth from RGB image using MiDaS."""
    h, w = img_rgb.shape[:2]
    inp = transform(img_rgb).to(device)
    with torch.no_grad():
        pred = midas(inp)  # [1,H',W'] or [1,1,H',W']
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=(h, w), mode="bicubic", align_corners=False)
        pred = pred.squeeze().float().cpu().numpy()  # HxW, float32
    return pred


def update_depth_maps_for_dataset(base_dir: str, device: torch.device, model_name: str = "DPT_Large"):
    """
    Update depth maps for a dataset using DPT_Large model.
    
    Args:
        base_dir: Base directory of the dataset (e.g., train1 or train2)
        device: Torch device
        model_name: DPT model name
    
    Returns:
        Number of depth maps updated
    """
    vi_dir = os.path.join(base_dir, 'vi')
    depth_dir = os.path.join(base_dir, 'depth', 'MSRS')
    depth_dir_alt = os.path.join(base_dir, 'depth')
    
    # Determine depth directory
    if os.path.exists(depth_dir):
        output_depth_dir = depth_dir
    elif os.path.exists(depth_dir_alt):
        output_depth_dir = depth_dir_alt
    else:
        # Create depth/MSRS directory
        output_depth_dir = depth_dir
        ensure_dir(Path(output_depth_dir))
    
    if not os.path.exists(vi_dir):
        print(f"  [WARN] VI directory not found: {vi_dir}")
        return 0
    
    # Load MiDaS model
    print(f"  [INFO] Loading MiDaS {model_name} model...")
    midas, transform = load_midas(model_name, device)
    
    # Get all VI images
    vi_path = Path(vi_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    vi_files = [f for f in vi_path.iterdir() if f.suffix.lower() in exts]
    
    if not vi_files:
        print(f"  [WARN] No VI images found in {vi_dir}")
        return 0
    
    print(f"  [INFO] Processing {len(vi_files)} VI images...")
    updated_count = 0
    
    for vi_file in tqdm(vi_files, desc=f"  Updating depth maps for {os.path.basename(base_dir)}"):
        try:
            # Read VI image (grayscale, but MiDaS needs RGB)
            vi_img = cv2.imread(str(vi_file), cv2.IMREAD_GRAYSCALE)
            if vi_img is None:
                print(f"  [WARN] Failed to read: {vi_file}")
                continue
            
            # Convert grayscale to RGB (3 channels)
            vi_rgb = cv2.cvtColor(vi_img, cv2.COLOR_GRAY2RGB)
            
            # Predict depth
            raw_depth = predict_depth(midas, transform, vi_rgb, device=device)
            depth01 = minmax_norm(raw_depth)  # Normalize to [0,1]
            
            # Save as 16-bit TIFF
            depth_file = Path(output_depth_dir) / vi_file.stem
            depth_file = depth_file.with_suffix(".tiff")
            ensure_dir(depth_file.parent)
            
            d16 = (depth01 * 65535.0).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_file), d16)
            updated_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {vi_file}: {e}")
            continue
    
    print(f"  [DONE] Updated {updated_count}/{len(vi_files)} depth maps in {base_dir}")
    return updated_count


def count_parameters(model):
    """Count trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def print_model_info(model):
    """Print detailed model architecture information."""
    print("\n" + "="*80)
    print("NETWORK ARCHITECTURE INFORMATION")
    print("="*80)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Print model structure
    print("\nModel Structure:")
    print("-" * 80)
    for name, module in model.named_children():
        if hasattr(module, '__len__'):
            print(f"  {name}: {len(module)} sub-modules")
        else:
            print(f"  {name}: {type(module).__name__}")
    
    # Check if depth branch is enabled
    if hasattr(model, 'dual_unet') and hasattr(model.dual_unet, 'use_depth'):
        print(f"\nDepth Branch: {'Enabled' if model.dual_unet.use_depth else 'Disabled'}")
        if model.dual_unet.use_depth:
            depth_params = count_parameters(model.dual_unet.depth_branch)
            # Count progressive fusion modules (vis_depth_fusions: vis2~vis5 only)
            fusion_params = 0
            if hasattr(model.dual_unet, 'vis_depth_fusions'):
                for fuse_module in model.dual_unet.vis_depth_fusions:
                    fusion_params += count_parameters(fuse_module)
                num_fusions = len(model.dual_unet.vis_depth_fusions)
                print(f"  Depth Branch parameters: {depth_params:,} ({depth_params/1e6:.2f}M)")
                print(f"  Progressive VIS-Depth Fusion (PaperHDGFFM, vis2~vis5 only): {fusion_params:,} ({fusion_params/1e6:.2f}M)")
                print(f"    - Fusion modules: {num_fusions} (vis1 skipped, no depth injection)")
            else:
                print(f"  Depth Branch parameters: {depth_params:,} ({depth_params/1e6:.2f}M)")
            
            # Count bottleneck fusion if exists
            if hasattr(model.dual_unet, 'bottleneck_fusion'):
                bottleneck_params = count_parameters(model.dual_unet.bottleneck_fusion)
                print(f"  Bottleneck IR-VIS Fusion (PaperHDGFFM) parameters: {bottleneck_params:,} ({bottleneck_params/1e6:.2f}M)")
    
    print("="*80 + "\n")


def print_loss_info():
    """Print detailed loss function information."""
    print("\n" + "="*80)
    print("LOSS FUNCTION INFORMATION")
    print("="*80)
    
    print(f"L_GradE parameters (USM-based):")
    print(f"  λ_gain: {utils.L_GRADE_LAMB}")
    print(f"  σ (sigma): {utils.L_GRADE_SIGMA1}")
    print(f"  sigma2: {utils.L_GRADE_SIGMA2} (unused)")
    
    print(f"\nLoss weights:")
    print(f"  α (L_gradE): {utils.LOSS_WEIGHT_GRADE}")
    print(f"  β (L_pixel): {utils.LOSS_WEIGHT_PIXEL}")
    print(f"  λ (L_depth): {utils.LOSS_WEIGHT_DEPTH}")
    
    print("="*80 + "\n")


def print_training_config():
    """Print training configuration."""
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Device: {utils.get_device()}")
    print(f"Batch Size: {utils.BATCH_SIZE}")
    print(f"Epochs: {utils.N_EPOCHS}")
    print(f"Learning Rate: {utils.LEARNING_RATE}")
    print(f"Optimizer: AdamW (betas=({utils.ADAM_BETA1}, {utils.ADAM_BETA2}), weight_decay={utils.ADAM_WEIGHT_DECAY})")
    print(f"Scheduler: StepLR (step_size={utils.SCHEDULER_STEP_SIZE}, gamma={utils.SCHEDULER_GAMMA})")
    print(f"Num Workers: {utils.NUM_WORKERS}")
    print(f"Pin Memory: Enabled")
    print(f"Gradient Accumulation Steps: {utils.GRADIENT_ACCUMULATION_STEPS} (effective batch size: {utils.BATCH_SIZE * utils.GRADIENT_ACCUMULATION_STEPS})")
    print(f"Seed: {utils.SEED}")
    print("\nNetwork Architecture:")
    print(f"  Base Channels: {utils.BASE_CH}")
    print(f"  Use Depth Branch: {utils.USE_DEPTH}")
    print(f"  Depth Branch Input Channels: {utils.DEPTH_BRANCH_IN_CH}")
    print(f"  Depth Branch Regress Depth: {utils.DEPTH_BRANCH_REGRESS_DEPTH}")
    print(f"  Fusion Heads: {utils.FUSION_HEADS}")
    print(f"  Fusion Tau: {utils.FUSION_TAU}")
    print(f"  Fusion With Norm: {utils.FUSION_WITH_NORM}")
    print("\nTraining Optimization:")
    # Determine normalization layer type based on batch size
    norm_type = "LayerNorm2d" if utils.BATCH_SIZE <= 4 else "BatchNorm2d"
    print(f"  Normalization Layer: {norm_type} (batch_size={utils.BATCH_SIZE})")
    # AMP status (will be enabled if CUDA is available)
    device = utils.get_device()
    amp_enabled = device.type == 'cuda'
    if amp_enabled:
        print(f"  Mixed Precision Training (AMP): Enabled (float16 with FFT@float32)")
    else:
        print(f"  Mixed Precision Training (AMP): Disabled (CPU or no CUDA)")
    print("\nDataset Configuration:")
    print(f"  Use Multi-Dataset: {utils.USE_MULTI_DATASET} (Disabled - using unified dataset)")
    print(f"  IR Dir: {utils.TRAIN_IR_DIR}")
    print(f"  VI Dir: {utils.TRAIN_VI_DIR}")
    print(f"  Depth Dir: {utils.TRAIN_DEPTH_DIR}")
    # Multi-dataset training info (COMMENTED OUT)
    # if utils.USE_MULTI_DATASET:
    #     print(f"  Train1 Epochs per Cycle: {utils.TRAIN1_EPOCHS_PER_CYCLE}")
    #     print(f"  Train2 Epochs per Cycle: {utils.TRAIN2_EPOCHS_PER_CYCLE}")
    #     print(f"  Cycle Length: {utils.CYCLE_LENGTH} ({utils.TRAIN1_EPOCHS_PER_CYCLE} train1 + {utils.TRAIN2_EPOCHS_PER_CYCLE} train2)")
    #     print(f"  Train1 Base Dir: {utils.TRAIN1_BASE_DIR}")
    #     print(f"  Train2 Base Dir: {utils.TRAIN2_BASE_DIR}")
    print("\nDepth Map Update:")
    print(f"  Use DPT for Depth Update: {utils.USE_DPT_FOR_DEPTH_UPDATE}")
    if utils.USE_DPT_FOR_DEPTH_UPDATE:
        print(f"  Depth Update Start Epoch: {utils.DEPTH_UPDATE_START_EPOCH}")
        print(f"  Depth Update Interval: {utils.DEPTH_UPDATE_INTERVAL}")
        print(f"  DPT Model Name: {utils.DPT_MODEL_NAME}")
    print("="*80)


class Config:
    """Configuration class to store all training parameters."""

    def __init__(self):
        self.parser = self._create_parser()
        self.args = None

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with training parameters."""
        parser = argparse.ArgumentParser(description='Image Fusion Network Training')

        # Dataset parameters
        parser.add_argument('--dataset', type=str, default='data', help='Dataset path')
        parser.add_argument('--input_nc', type=int, default=1, help='Input image channels')
        parser.add_argument('--output_nc', type=int, default=1, help='Output image channels')

        # Training parameters
        parser.add_argument('--batchSize', type=int, default=4, help='Training batch size')
        parser.add_argument('--testBatchSize', type=int, default=1, help='Testing batch size')
        parser.add_argument('--nEpochs', type=int, default=100, help='Number of epochs')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
        parser.add_argument('--alpha', type=float, default=0.25, help='Alpha parameter for loss')
        parser.add_argument('--lamb', type=int, default=150, help='Lambda weight for L1 loss')

        # Network parameters
        parser.add_argument('--ngf', type=int, default=64, help='Generator filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='Discriminator filters in first conv layer')

        # System parameters
        parser.add_argument('--cuda', action='store_true', help='Use CUDA')
        parser.add_argument('--threads', type=int, default=0, help='Number of threads for data loader')
        parser.add_argument('--seed', type=int, default=123, help='Random seed')
        parser.add_argument('--ema_decay', type=float, default=0.9, help='EMA decay rate')

        return parser

    def parse_args(self):
        """Parse command line arguments."""
        self.args = self.parser.parse_args()
        return self.args


class Trainer:
    """Main trainer class implementing the training pipeline."""

    def __init__(self):
        self.device = self._setup_device()
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        # Initialize GradScaler for mixed precision training (AMP)
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        # Initialize dataloader for epoch 1
        self.dataloader = self._setup_dataloader(epoch=1)
        # Default: train both branches (depth branch always exists)
        if hasattr(self.model, 'dual_unet') and self.model.dual_unet.use_depth:
            self._unfreeze_depth_branch()
        
        # Initialize loss history for plotting
        self.loss_history = {
            'total': [],
            'gradE': [],
            'pixel': [],
            'mask': [],
            'depth': [],
            'epoch': []
        }
        
        # Print detailed information
        print_training_config()
        print_model_info(self.model)
        print_loss_info()

    def _setup_device(self) -> torch.device:
        """Setup device from utils policy."""
        device = utils.get_device()
        torch.manual_seed(utils.SEED)
        if device.type == 'cuda':
            torch.cuda.manual_seed(utils.SEED)
        cudnn.benchmark = True
        print(f"Using device: {device}")
        return device

    def _setup_model(self) -> nn.Module:
        """Initialize and setup the network model."""
        model = Net().to(self.device)
        return model

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup AdamW optimizer (more efficient than Adam)."""
        return optim.AdamW(
            self.model.parameters(),
            lr=utils.LEARNING_RATE,
            betas=(utils.ADAM_BETA1, utils.ADAM_BETA2),
            weight_decay=utils.ADAM_WEIGHT_DECAY
        )

    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=utils.SCHEDULER_STEP_SIZE,
            gamma=utils.SCHEDULER_GAMMA
        )

    def _setup_dataloader(self, epoch: int = 1) -> DataLoader:
        """Setup data loader with dataset paths based on epoch."""
        # Get dataset paths based on epoch
        ir_dir, vi_dir, depth_dir = utils.get_dataset_paths(epoch)
        
        dataset = fusiondata(
            ir_dir=ir_dir, 
            vi_dir=vi_dir,
            depth_dir=depth_dir
        )
        
        print(f"Dataset loaded for epoch {epoch}:")
        print(f"  IR dir: {ir_dir}")
        print(f"  VI dir: {vi_dir}")
        print(f"  Depth dir: {depth_dir if depth_dir else 'None'}")
        print(f"  Dataset size: {len(dataset)} pairs")
        
        return DataLoader(
            dataset=dataset,
            num_workers=utils.NUM_WORKERS,
            batch_size=utils.BATCH_SIZE,
            shuffle=True,
            pin_memory=True,  # Enable pin memory for faster CPU-GPU data transfer
            persistent_workers=True if utils.NUM_WORKERS > 0 else False  # Keep workers alive between epochs
        )
    
    def _reload_dataloader(self, epoch: int) -> None:
        """Reload dataloader with new dataset paths for the given epoch."""
        self.dataloader = self._setup_dataloader(epoch)
    
    def _update_depth_maps(self, epoch: int) -> None:
        """
        Update depth maps for train1 and train2 using DPT_Large model.
        Called at epochs 40, 44, 48, ...
        """
        if not utils.USE_DPT_FOR_DEPTH_UPDATE:
            return
        
        print(f"\n{'='*80}")
        print(f"[Epoch {epoch}] Updating depth maps using DPT_Large model...")
        print(f"{'='*80}\n")
        
        # Update depth maps for unified training dataset
        print(f"Updating depth maps for training dataset...")
        # Extract base directory from TRAIN_VI_DIR (assuming structure: .../train/vi)
        base_dir = os.path.dirname(os.path.dirname(utils.TRAIN_VI_DIR))
        count = update_depth_maps_for_dataset(
            base_dir,
            self.device,
            model_name=utils.DPT_MODEL_NAME
        )
        
        # Multi-dataset depth update (COMMENTED OUT)
        # # Update depth maps for train1
        # print(f"Updating depth maps for train1...")
        # count1 = update_depth_maps_for_dataset(
        #     utils.TRAIN1_BASE_DIR,
        #     self.device,
        #     model_name=utils.DPT_MODEL_NAME
        # )
        # 
        # # Update depth maps for train2
        # print(f"\nUpdating depth maps for train2...")
        # count2 = update_depth_maps_for_dataset(
        #     utils.TRAIN2_BASE_DIR,
        #     self.device,
        #     model_name=utils.DPT_MODEL_NAME
        # )
        
        print(f"\n{'='*80}")
        print(f"[DONE] Depth maps updated: {count} images")
        print(f"{'='*80}\n")
        
        # Reload dataloader to use updated depth maps
        print(f"Reloading dataloader with updated depth maps...")
        self._reload_dataloader(epoch)
    
    def _freeze_depth_branch(self) -> None:
        """Freeze depth branch parameters (depth_branch and vis_depth_fusions)."""
        if not hasattr(self.model, 'dual_unet') or not self.model.dual_unet.use_depth:
            return
        
        # Freeze depth_branch parameters
        for param in self.model.dual_unet.depth_branch.parameters():
            param.requires_grad = False
        
        # Freeze vis_depth_fusions parameters
        if hasattr(self.model.dual_unet, 'vis_depth_fusions'):
            for fusion_module in self.model.dual_unet.vis_depth_fusions:
                for param in fusion_module.parameters():
                    param.requires_grad = False
        
        print("  Depth branch parameters frozen (only fusion main branch will be trained)")
    
    def _unfreeze_depth_branch(self) -> None:
        """Unfreeze depth branch parameters (depth_branch and vis_depth_fusions)."""
        if not hasattr(self.model, 'dual_unet') or not self.model.dual_unet.use_depth:
            return
        
        # Unfreeze depth_branch parameters
        for param in self.model.dual_unet.depth_branch.parameters():
            param.requires_grad = True
        
        # Unfreeze vis_depth_fusions parameters
        if hasattr(self.model.dual_unet, 'vis_depth_fusions'):
            for fusion_module in self.model.dual_unet.vis_depth_fusions:
                for param in fusion_module.parameters():
                    param.requires_grad = True
        
        print("  Depth branch parameters unfrozen (both branches will be trained)")

    def train_epoch(self, epoch: int) -> None:
        """Train for one epoch."""
        self.model.train()
        print(f"\n[Epoch {epoch}] Training mode: Both Branches (Depth Branch + Fusion Main Branch)")
        
        t0 = time.time()
        rolling = {"loss": 0.0, "gradE": 0.0, "pixel": 0.0, "mask": 0.0, "depth": 0.0}
        
        # Loss weights from utils
        alpha = utils.LOSS_WEIGHT_GRADE
        beta = utils.LOSS_WEIGHT_PIXEL
        gamma = utils.LOSS_WEIGHT_MASK
        lambda_depth = utils.LOSS_WEIGHT_DEPTH
        
        # Gradient accumulation settings
        accumulation_steps = utils.GRADIENT_ACCUMULATION_STEPS
        effective_batch_size = utils.BATCH_SIZE * accumulation_steps
        print(f"  Gradient Accumulation: {accumulation_steps} steps (effective batch size: {effective_batch_size})")
        
        # Initialize optimizer gradients at the start of epoch
        self.optimizer.zero_grad()
        
        # Track total batches for final gradient update
        total_batches = len(self.dataloader)

        for batch_idx, batch_data in enumerate(self.dataloader, 1):
            # Handle both cases: with and without depth
            if self.dataloader.dataset.has_depth:
                imgA, imgB, d_teacher = batch_data
                d_teacher = (d_teacher / 255).to(self.device)  # Normalize depth to [0,1]
            else:
                imgA, imgB = batch_data
                d_teacher = None
            
            # Normalize images to [0,1] range and move to device
            imgA = (imgA / 255).to(self.device)
            imgB = (imgB / 255).to(self.device)
            
            # Mixed precision training (AMP): use autocast for forward pass
            # FFT operations in FourierMix2D will automatically use float32 via @custom_fwd
            with autocast(dtype=torch.float16, enabled=(self.scaler is not None)):
                # Forward pass (returns: fused_image, zero_loss, d_hat)
                fused_image, _, d_hat = self.model(imgA, imgB)

                # (Optional) extract Y channel once and reuse (kept for compatibility)
                imgA_Y = imgA[:, :1, :, :]
                imgB_Y = imgB[:, :1, :, :]
                fused_Y = fused_image[:, :1, :, :]

                # Calculate image fusion losses: Ltotal = α * LgradE + β * Lpixel + γ * Lmask
                loss_gradE = L_GradE(
                    imgA, imgB, fused_image,
                    lamb=utils.L_GRADE_LAMB,
                    sigma1=utils.L_GRADE_SIGMA1,
                    sigma2=utils.L_GRADE_SIGMA2,
                )
                loss_pixel = L_Int(imgA, imgB, fused_image)
                loss_mask = L_Mask(imgA, imgB, fused_image, sigma_low=utils.L_MASK_SIGMA_LOW)

                loss_fusion = alpha * loss_gradE + beta * loss_pixel + gamma * loss_mask

                # Depth distillation loss (if depth branch is enabled and teacher depth is available)
                loss_depth = torch.tensor(0.0, device=self.device, dtype=fused_image.dtype)
                if d_hat is not None and lambda_depth > 0:
                    if d_teacher is not None:
                        # Option 1: Supervised depth loss using teacher depth
                        d_hat_norm = normalize_per_image(d_hat)
                        d_teacher_norm = normalize_per_image(d_teacher)
                        if d_hat_norm.shape != d_teacher_norm.shape:
                            d_hat_norm = F.interpolate(
                                d_hat_norm,
                                size=d_teacher_norm.shape[2:],
                                mode='bilinear',
                                align_corners=False,
                            )
                        loss_depth = F.mse_loss(d_hat_norm, d_teacher_norm)
                    else:
                        # Option 2: Self-supervised depth loss (depth smoothness)
                        depth_grad_x = torch.abs(d_hat[:, :, :, :-1] - d_hat[:, :, :, 1:])
                        depth_grad_y = torch.abs(d_hat[:, :, :-1, :] - d_hat[:, :, 1:, :])
                        vis_grad_x = torch.abs(imgB[:, :, :, :-1] - imgB[:, :, :, 1:])
                        vis_grad_y = torch.abs(imgB[:, :, :-1, :] - imgB[:, :, 1:, :])
                        smoothness_loss = (
                            torch.mean(depth_grad_x * torch.exp(-vis_grad_x)) +
                            torch.mean(depth_grad_y * torch.exp(-vis_grad_y))
                        )
                        loss_depth = smoothness_loss * utils.DEPTH_SMOOTHNESS_SCALE

                # Total loss
                total_loss = loss_fusion + lambda_depth * loss_depth

            # Total loss (divide by accumulation_steps for gradient accumulation)
            # Note: loss is already in the correct dtype from autocast, no need to convert
            loss = total_loss / accumulation_steps

            # Backward pass with gradient accumulation
            if self.scaler is not None:
                # Use GradScaler for mixed precision training
                self.scaler.scale(loss).backward()
            else:
                # CPU or no AMP: standard backward pass
                loss.backward()
            
            # Update optimizer every accumulation_steps batches, or at the last batch
            if batch_idx % accumulation_steps == 0 or batch_idx == total_batches:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Accumulate losses (multiply by accumulation_steps to get true loss values for logging)
            true_loss = loss * accumulation_steps  # Scale back for logging
            rolling["loss"] += true_loss.item()
            rolling["gradE"] += loss_gradE.item()
            rolling["pixel"] += loss_pixel.item()
            rolling["mask"] += loss_mask.item()
            rolling["depth"] += loss_depth.item()

            # Print progress every N batches (adjusted for gradient accumulation)
            print_interval = max(10, accumulation_steps)  # At least every accumulation_steps batches
            if batch_idx % print_interval == 0:
                dt = time.time() - t0
                avg_loss = rolling["loss"] / print_interval
                avg_gradE = rolling["gradE"] / print_interval
                avg_pixel = rolling["pixel"] / print_interval
                avg_mask = rolling["mask"] / print_interval
                avg_depth = rolling["depth"] / print_interval
                
                # Detailed loss breakdown
                weighted_gradE = alpha * avg_gradE
                weighted_pixel = beta * avg_pixel
                weighted_mask = gamma * avg_mask
                weighted_depth = lambda_depth * avg_depth
                
                print(f'\n[Epoch {epoch}/{utils.N_EPOCHS}, Batch {batch_idx}] Time: {dt:.2f}s')
                print(f'  Total Loss: {avg_loss:.6f} (effective batch: {effective_batch_size})')
                print(f'  ┌─ L_gradE:  {avg_gradE:.6f} × {alpha:.1f} = {weighted_gradE:.6f}')
                print(f'  ├─ L_pixel:  {avg_pixel:.6f} × {beta:.1f} = {weighted_pixel:.6f}')
                print(f'  └─ L_depth:  {avg_depth:.6f} × {lambda_depth:.1f} = {weighted_depth:.6f}')
                print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.2e}')
                
                rolling = {"loss": 0.0, "gradE": 0.0, "pixel": 0.0, "mask": 0.0, "depth": 0.0}
                t0 = time.time()
        
        # Calculate epoch average losses
        num_batches = len(self.dataloader)
        epoch_avg_loss = rolling["loss"] / num_batches if num_batches > 0 else 0.0
        epoch_avg_gradE = rolling["gradE"] / num_batches if num_batches > 0 else 0.0
        epoch_avg_pixel = rolling["pixel"] / num_batches if num_batches > 0 else 0.0
        epoch_avg_mask = rolling["mask"] / num_batches if num_batches > 0 else 0.0
        epoch_avg_depth = rolling["depth"] / num_batches if num_batches > 0 else 0.0
        
        # Record loss history
        self.loss_history['epoch'].append(epoch)
        self.loss_history['total'].append(epoch_avg_loss)
        self.loss_history['gradE'].append(epoch_avg_gradE)
        self.loss_history['pixel'].append(epoch_avg_pixel)
        self.loss_history['mask'].append(epoch_avg_mask)
        self.loss_history['depth'].append(epoch_avg_depth)
        
        # Save checkpoint and plot loss curve
        self._save_checkpoint(epoch)
        self._plot_loss_curve()

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(utils.MODEL_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _plot_loss_curve(self) -> None:
        """Plot and save loss curves."""
        if len(self.loss_history['epoch']) == 0:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Loss Curves', fontsize=16, fontweight='bold')
        
        # Total loss
        axes[0, 0].plot(self.loss_history['epoch'], self.loss_history['total'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # L_gradE
        axes[0, 1].plot(self.loss_history['epoch'], self.loss_history['gradE'], 'r-', linewidth=2)
        axes[0, 1].set_title('L_gradE (Gradient Enhancement Loss)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # L_pixel
        axes[0, 2].plot(self.loss_history['epoch'], self.loss_history['pixel'], 'g-', linewidth=2)
        axes[0, 2].set_title('L_pixel (Intensity Loss)', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        
        # L_depth
        axes[1, 1].plot(self.loss_history['epoch'], self.loss_history['depth'], 'c-', linewidth=2)
        axes[1, 1].set_title('L_depth (Depth Loss)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        # All losses together
        axes[1, 2].plot(self.loss_history['epoch'], self.loss_history['total'], 'b-', label='Total', linewidth=2)
        axes[1, 2].plot(self.loss_history['epoch'], self.loss_history['gradE'], 'r-', label='L_gradE', linewidth=1.5, alpha=0.7)
        axes[1, 2].plot(self.loss_history['epoch'], self.loss_history['pixel'], 'g-', label='L_pixel', linewidth=1.5, alpha=0.7)
        axes[1, 2].plot(self.loss_history['epoch'], self.loss_history['depth'], 'c-', label='L_depth', linewidth=1.5, alpha=0.7)
        axes[1, 2].set_title('All Losses', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend(loc='upper right')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        loss_curve_path = os.path.join(utils.LOSS_DIR, 'loss_curves.png')
        plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Loss curves saved: {loss_curve_path}")

    def train(self) -> None:
        """Main training loop."""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")
        
        for epoch in range(1, utils.N_EPOCHS + 1):
            # Check if we need to update depth maps (epoch >= 40 and every 4 epochs: 40, 44, 48, ...)
            should_update_depth = (epoch >= utils.DEPTH_UPDATE_START_EPOCH and 
                                  (epoch - utils.DEPTH_UPDATE_START_EPOCH) % utils.DEPTH_UPDATE_INTERVAL == 0)
            
            if should_update_depth:
                # Update depth maps using DPT_Large
                self._update_depth_maps(epoch)
                # Reload dataloader to use updated depth maps
                self._reload_dataloader(epoch)
            
            # Multi-dataset training code (COMMENTED OUT - using unified dataset)
            # if utils.USE_MULTI_DATASET:
            #     # Calculate position in cycle (1-indexed)
            #     cycle_pos = ((epoch - 1) % utils.CYCLE_LENGTH) + 1
            #     
            #     # Determine which dataset should be used for this epoch
            #     # If cycle_pos > TRAIN1_EPOCHS_PER_CYCLE: use train2
            #     # Otherwise: use train1
            #     if cycle_pos > utils.TRAIN1_EPOCHS_PER_CYCLE:
            #         expected_dataset = 'train2'
            #     else:
            #         expected_dataset = 'train1'
            #     
            #     # Reload dataloader and adjust training mode if dataset needs to change
            #     if expected_dataset != self.current_dataset:
            #         cycle_num = (epoch - 1) // utils.CYCLE_LENGTH + 1
            #         print(f"\n[Epoch {epoch}] Cycle {cycle_num}, Position {cycle_pos}/{utils.CYCLE_LENGTH}")
            #         print(f"  Switching dataset from {self.current_dataset} to {expected_dataset}...")
            #         
            #         # Reload dataloader
            #         self._reload_dataloader(epoch)
            #         
            #         # Adjust depth branch training status based on dataset
            #         if expected_dataset == 'train2':
            #             # train2: Freeze depth branch, only train fusion main branch
            #             self._freeze_depth_branch()
            #         else:
            #             # train1: Unfreeze depth branch, train both branches
            #             self._unfreeze_depth_branch()
            #         
            #         self.current_dataset = expected_dataset
            
            self.train_epoch(epoch)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f'\n[Epoch {epoch}/{utils.N_EPOCHS} Completed] Learning Rate: {current_lr:.2e}')
            print("-" * 80)


def main():
    """Main function."""
    # Initialize trainer (will print config/model/loss info in __init__)
    trainer = Trainer()
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()