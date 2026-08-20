#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Fusion Network Training Script
This script implements training pipeline for an image fusion network.
"""

import argparse
import csv
import os
import re
import time
from typing import Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Tkinter thread issues
import matplotlib.pyplot as plt

from moudle import fusiondata
#from net_no_crossattn_ablation import Net
#from net_no_selfattn_ablation import Net
#from net_depth_ir_only_ablation import Net
from net import Net
#from net_depth_vis_only_ablation import Net
#from net_crossattn_concat_ablation import Net
#from net_enhanceddown_cnn_ablation import Net
#from net_enhanceddown_fnet_ablation import Net
#from net_paperhdgffm_concat_ablation import Net
#from net_no_depth_ablation import Net
from loss import L_GradE, L_Int
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


def load_depth_anything(model_name="depth-anything/Depth-Anything-V2-Small-hf", device="cuda"):
    """Load Depth Anything / Depth Anything V2 from the local Transformers directory."""
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise ImportError(
            "Depth Anything requires the 'transformers' package. "
            "Install it with: pip install transformers"
        ) from exc

    local_model_dir = Path(utils.DPT_LOCAL_MODEL_DIR)
    if not local_model_dir.exists():
        raise FileNotFoundError(
            f"Local Depth Anything model not found: {local_model_dir}\n"
            f"Download it first with:\n"
            f"  python download_depth_anything.py\n"
            f"Training loads Depth Anything from local files only to avoid network failures."
        )

    processor = AutoImageProcessor.from_pretrained(local_model_dir, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(local_model_dir, local_files_only=True)
    model.to(device).eval()
    return model, processor


def predict_depth(depth_model, processor, img_rgb, device="cuda"):
    """Predict depth from an RGB image using Depth Anything."""
    h, w = img_rgb.shape[:2]
    inputs = processor(images=img_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = depth_model(**inputs)
        pred = outputs.predicted_depth  # [1,H',W'] or [1,1,H',W']
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=(h, w), mode="bicubic", align_corners=False)
        pred = pred.squeeze().float().cpu().numpy()  # HxW, float32
    return pred


def update_depth_maps_for_dataset(
    ir_dir: str,
    vi_dir: str,
    depth_dir: str,
    device: torch.device,
    fusion_model: nn.Module,
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
):
    """
    Update teacher depth maps using current fused images.

    Flow:
        IR + VIS -> current fusion model -> fused image -> Depth Anything -> teacher depth
    
    Args:
        ir_dir: Training IR directory
        vi_dir: Training VIS directory
        depth_dir: Output teacher depth directory
        device: Torch device
        fusion_model: Current fusion network used to generate fused images
        model_name: HuggingFace Depth Anything model name
    
    Returns:
        Number of depth maps updated
    """
    if not os.path.exists(ir_dir):
        print(f"  [WARN] IR directory not found: {ir_dir}")
        return 0
    if not os.path.exists(vi_dir):
        print(f"  [WARN] VI directory not found: {vi_dir}")
        return 0

    output_depth_dir = depth_dir
    ensure_dir(Path(output_depth_dir))
    
    # Load Depth Anything model
    print(f"  [INFO] Loading Depth Anything model from local dir: {utils.DPT_LOCAL_MODEL_DIR}")
    depth_model, processor = load_depth_anything(model_name, device)

    # Match IR/VIS pairs by basename, consistent with training dataset logic.
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def list_images_by_stem(folder: str):
        files = [f for f in Path(folder).iterdir() if f.suffix.lower() in exts]
        return {f.stem: f for f in files}

    def sort_key(stem: str):
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    ir_files = list_images_by_stem(ir_dir)
    vi_files = list_images_by_stem(vi_dir)
    common_stems = sorted(set(ir_files.keys()) & set(vi_files.keys()), key=sort_key)

    if not common_stems:
        print(f"  [WARN] No matched IR/VIS pairs found in {ir_dir} and {vi_dir}")
        return 0

    print(f"  [INFO] Processing {len(common_stems)} fused IR/VIS images...")
    updated_count = 0

    was_training = fusion_model.training
    fusion_model.eval()

    for stem in tqdm(common_stems, desc=f"  Updating fused-depth maps for {Path(vi_dir).parent.name}"):
        try:
            ir_file = ir_files[stem]
            vi_file = vi_files[stem]

            ir_img = cv2.imread(str(ir_file), cv2.IMREAD_GRAYSCALE)
            vi_img = cv2.imread(str(vi_file), cv2.IMREAD_GRAYSCALE)
            if ir_img is None or vi_img is None:
                print(f"  [WARN] Failed to read pair: IR={ir_file}, VI={vi_file}")
                continue

            ir_tensor = torch.from_numpy(ir_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            vi_tensor = torch.from_numpy(vi_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                fused_image, _, _ = fusion_model(ir_tensor, vi_tensor)

            fused_gray = fused_image.squeeze().detach().float().clamp(0, 1).cpu().numpy()
            fused_u8 = (fused_gray * 255.0).round().astype(np.uint8)
            
            # Depth Anything expects RGB input.
            fused_rgb = cv2.cvtColor(fused_u8, cv2.COLOR_GRAY2RGB)
            
            # Predict depth from the current fused image.
            raw_depth = predict_depth(depth_model, processor, fused_rgb, device=device)
            depth01 = minmax_norm(raw_depth)  # Normalize to [0,1]
            
            # Save as 16-bit TIFF
            depth_file = Path(output_depth_dir) / stem
            depth_file = depth_file.with_suffix(".tiff")
            ensure_dir(depth_file.parent)
            
            d16 = (depth01 * 65535.0).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_file), d16)
            updated_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Failed to process pair '{stem}': {e}")
            continue

    if was_training:
        fusion_model.train()
    
    print(f"  [DONE] Updated {updated_count}/{len(common_stems)} fused-depth maps in {output_depth_dir}")
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
    cosine_t_max = max(1, utils.N_EPOCHS - utils.SCHEDULER_START_EPOCH + 1)
    cosine_eta_min = getattr(utils, 'SCHEDULER_ETA_MIN', 0.0)
    print(f"Optimizer: AdamW (betas=({utils.ADAM_BETA1}, {utils.ADAM_BETA2}), weight_decay={utils.ADAM_WEIGHT_DECAY})")
    print(
        "Scheduler: CosineAnnealingLR "
        f"(start_epoch={utils.SCHEDULER_START_EPOCH}, T_max={cosine_t_max}, eta_min={cosine_eta_min})"
    )
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
    # AMP status (disabled for numerical stability)
    print(f"  Mixed Precision Training (AMP): Disabled (for stability)")
    print("\nAlgorithm 1 Schedule:")
    print(f"  Stage 1: {utils.PRETRAIN_EPOCHS} epochs (F only)")
    print(f"  Stage 2: {utils.NUM_SYNERGY_CYCLES} cycles × ({utils.FUSION_EPOCHS_PER_CYCLE} F-only + {utils.JOINT_EPOCHS_PER_CYCLE} F+D) epochs")
    print(f"  D_sink: top-{utils.SINK_TOPK} ir5 channels, calibrated after Stage 1")
    print("\nDataset Configuration:")
    print("  Unified MSRS split deterministically into Dtrain1 and Dtrain2")
    print(f"  IR Dir: {utils.TRAIN_IR_DIR}")
    print(f"  VI Dir: {utils.TRAIN_VI_DIR}")
    print(f"  Depth Dir: {utils.TRAIN_DEPTH_DIR}")
    print("\nTeacher Depth Update:")
    print(f"  Use Depth Anything for Depth Update: {utils.USE_DPT_FOR_DEPTH_UPDATE}")
    if utils.USE_DPT_FOR_DEPTH_UPDATE:
        print("  Generated after Stage 1 and refreshed after every Stage 2 cycle")
        print(f"  Depth Anything Model Name: {utils.DPT_MODEL_NAME}")
        print(f"  Depth Anything Local Dir: {utils.DPT_LOCAL_MODEL_DIR}")
        print(f"  Updated Depth Output Dir: {utils.TRAIN_UPDATED_DEPTH_DIR}")
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
        utils.ensure_dirs()
        self.device = self._setup_device()
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        # AMP disabled for stability (was: GradScaler() if CUDA available)
        self.scaler = None
        # Stage 1 has no teacher labels; they are generated from the pre-trained F.
        self.current_depth_dir = None
        self.dataloader = self._setup_dataloader(epoch=1)
        self.split_indices = None
        if hasattr(self.model, 'dual_unet') and self.model.dual_unet.use_depth:
            self._freeze_depth_branch()
        
        # Initialize loss history for plotting
        self.loss_history = {
            'total': [],
            'gradE': [],
            'pixel': [],
            'depth': [],
            'epoch': []
        }
        self.depth_quality_history = {
            'epoch': [],
            'mse': [],
            'mae': [],
            'rmse': [],
            'psnr': []
        }
        self.start_epoch = 1
        self._resume_if_requested()
        
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
        t_max = max(1, utils.N_EPOCHS - utils.SCHEDULER_START_EPOCH + 1)
        eta_min = getattr(utils, 'SCHEDULER_ETA_MIN', 0.0)
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=eta_min
        )

    def _setup_dataloader(self, epoch: int = 1, split: str = None) -> DataLoader:
        """Setup data loader with dataset paths based on epoch."""
        # Get dataset paths based on epoch
        ir_dir, vi_dir, _ = utils.get_dataset_paths(epoch)
        depth_dir = self.current_depth_dir
        
        dataset = fusiondata(
            ir_dir=ir_dir, 
            vi_dir=vi_dir,
            depth_dir=depth_dir
        )
        
        print(f"Dataset loaded for epoch {epoch}:")
        print(f"  IR dir: {ir_dir}")
        print(f"  VI dir: {vi_dir}")
        print(f"  Depth dir: {depth_dir if depth_dir else 'None'}")
        if depth_dir == utils.TRAIN_UPDATED_DEPTH_DIR:
            print(f"  Depth source: updated fused-depth teachers")
        else:
            print(f"  Depth source: original teachers")
        print(f"  Dataset size: {len(dataset)} pairs")
        
        if split is not None:
            if self.split_indices is None:
                indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(utils.SEED)).tolist()
                midpoint = len(indices) // 2
                self.split_indices = {"train1": indices[:midpoint], "train2": indices[midpoint:]}
                print(f"  Algorithm 1 split: Dtrain1={midpoint}, Dtrain2={len(indices) - midpoint}")
            dataset = Subset(dataset, self.split_indices[split])
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

    def _set_dataloader_for_split(self, epoch: int, split: str) -> None:
        self.dataloader = self._setup_dataloader(epoch, split=split)

    def _apply_sink_dims(self, sink_dims: torch.Tensor) -> None:
        """Install calibrated, fixed ir5 sink channels in bottleneck VAR."""
        sink_dims = sink_dims.detach().to(dtype=torch.long, device=self.device)
        utils.VAR_SINK_DIMS = sink_dims.cpu().tolist()
        bottleneck = self.model.dual_unet.bottleneck_fusion
        bottleneck.var_sink_dims = sink_dims
        head_dim = bottleneck.fuse.d
        mask = torch.zeros(bottleneck.fuse.h, head_dim, dtype=torch.bool, device=self.device)
        for index in sink_dims.tolist():
            mask[min(bottleneck.fuse.h - 1, index // head_dim), index % head_dim] = True
        bottleneck.var_sink_mask_heads = mask
        bottleneck.enable_var = True

    def _calibrate_sink_channels(self) -> None:
        """Identify D_sink as top-K RMS-normalized ir5 activation channels."""
        dataset = fusiondata(utils.TRAIN_IR_DIR, utils.TRAIN_VI_DIR, depth_dir=None)
        loader = DataLoader(dataset, batch_size=utils.BATCH_SIZE, shuffle=False,
                            num_workers=utils.NUM_WORKERS, pin_memory=True)
        sums, count = None, 0
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for ir, _ in loader:
                ir = (ir / 255.0).to(self.device)
                net = self.model.dual_unet
                ir5 = net.ir_down4(net.ir_down3(net.ir_down2(net.ir_down1(net.ir_inc(ir)))))
                tokens = ir5.permute(0, 2, 3, 1).reshape(-1, ir5.shape[1])
                tokens = tokens / torch.sqrt(torch.mean(tokens ** 2, dim=1, keepdim=True) + 1e-6)
                values = tokens.abs().sum(dim=0).double().cpu()
                sums = values if sums is None else sums + values
                count += tokens.shape[0]
        if was_training:
            self.model.train()
        if sums is None:
            raise RuntimeError("Sink calibration found no paired MSRS samples.")
        sink_dims = torch.topk(sums / count, min(utils.SINK_TOPK, sums.numel())).indices
        self._apply_sink_dims(sink_dims)
        print(f"Calibrated D_sink: {utils.VAR_SINK_DIMS}")
    
    def _update_depth_maps(self, epoch: int) -> None:
        """
        Update depth maps using Depth Anything.
        Called at epochs 40, 44, 48, ...
        """
        if not utils.USE_DPT_FOR_DEPTH_UPDATE:
            return
        
        print(f"\n{'='*80}")
        print(f"[Epoch {epoch}] Updating depth maps using Depth Anything...")
        print(f"{'='*80}\n")
        
        # Update depth maps for unified training dataset using current fused images.
        print(f"Updating depth maps for training dataset from current fused images...")
        count = update_depth_maps_for_dataset(
            ir_dir=utils.TRAIN_IR_DIR,
            vi_dir=utils.TRAIN_VI_DIR,
            depth_dir=utils.TRAIN_UPDATED_DEPTH_DIR,
            device=self.device,
            fusion_model=self.model,
            model_name=utils.DPT_MODEL_NAME,
        )
        self.current_depth_dir = utils.TRAIN_UPDATED_DEPTH_DIR
        
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
        """Freeze D only; all fusion components remain trainable as part of F."""
        if not hasattr(self.model, 'dual_unet') or not self.model.dual_unet.use_depth:
            return
        
        # Freeze depth_branch parameters
        for param in self.model.dual_unet.depth_branch.parameters():
            param.requires_grad = False
        
        print("  Depth branch frozen; fusion branch F remains trainable")
    
    def _unfreeze_depth_branch(self) -> None:
        """Activate D for joint F+D optimization."""
        if not hasattr(self.model, 'dual_unet') or not self.model.dual_unet.use_depth:
            return
        
        # Unfreeze depth_branch parameters
        for param in self.model.dual_unet.depth_branch.parameters():
            param.requires_grad = True
        
        print("  Depth branch parameters unfrozen (both branches will be trained)")

    def train_epoch(self, epoch: int, mode: str, use_depth_loss: bool) -> None:
        """Train for one epoch."""
        self.model.train()
        print(f"\n[Epoch {epoch}/{utils.N_EPOCHS}] {mode}")
        
        t0 = time.time()
        rolling = {"loss": 0.0, "gradE": 0.0, "pixel": 0.0, "depth": 0.0}
        epoch_totals = {"loss": 0.0, "gradE": 0.0, "pixel": 0.0, "depth": 0.0}
        depth_quality_totals = {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "psnr": 0.0}
        depth_quality_batches = 0
        
        # Loss weights from utils
        alpha = utils.LOSS_WEIGHT_GRADE
        beta = utils.LOSS_WEIGHT_PIXEL
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
            if len(batch_data) == 3:
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
            depth_metric_mse = None
            depth_metric_mae = None
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
                loss_fusion = alpha * loss_gradE + beta * loss_pixel

                # Depth distillation loss (if depth branch is enabled and teacher depth is available)
                loss_depth = torch.tensor(0.0, device=self.device, dtype=fused_image.dtype)
                if use_depth_loss and d_hat is not None and lambda_depth > 0:
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
                        depth_diff = d_hat_norm.float() - d_teacher_norm.float()
                        depth_metric_mse = torch.mean(depth_diff ** 2)
                        depth_metric_mae = torch.mean(torch.abs(depth_diff))
                        loss_depth = depth_metric_mse.to(dtype=fused_image.dtype)
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
            true_loss_value = true_loss.item()
            gradE_value = loss_gradE.item()
            pixel_value = loss_pixel.item()
            depth_value = loss_depth.item()

            rolling["loss"] += true_loss_value
            rolling["gradE"] += gradE_value
            rolling["pixel"] += pixel_value
            rolling["depth"] += depth_value

            epoch_totals["loss"] += true_loss_value
            epoch_totals["gradE"] += gradE_value
            epoch_totals["pixel"] += pixel_value
            epoch_totals["depth"] += depth_value

            if depth_metric_mse is not None and depth_metric_mae is not None:
                mse_value = depth_metric_mse.item()
                mae_value = depth_metric_mae.item()
                rmse_value = mse_value ** 0.5
                psnr_value = 10.0 * np.log10(1.0 / max(mse_value, 1e-12))
                depth_quality_totals["mse"] += mse_value
                depth_quality_totals["mae"] += mae_value
                depth_quality_totals["rmse"] += rmse_value
                depth_quality_totals["psnr"] += psnr_value
                depth_quality_batches += 1

            # Print progress every N batches (adjusted for gradient accumulation)
            print_interval = max(10, accumulation_steps)  # At least every accumulation_steps batches
            if batch_idx % print_interval == 0:
                dt = time.time() - t0
                avg_loss = rolling["loss"] / print_interval
                avg_gradE = rolling["gradE"] / print_interval
                avg_pixel = rolling["pixel"] / print_interval
                avg_depth = rolling["depth"] / print_interval
                
                # Detailed loss breakdown
                weighted_gradE = alpha * avg_gradE
                weighted_pixel = beta * avg_pixel
                weighted_depth = lambda_depth * avg_depth
                
                print(f'\n[Epoch {epoch}/{utils.N_EPOCHS}, Batch {batch_idx}] Time: {dt:.2f}s')
                print(f'  Total Loss: {avg_loss:.6f} (effective batch: {effective_batch_size})')
                print(f'  ┌─ L_gradE:  {avg_gradE:.6f} × {alpha:.1f} = {weighted_gradE:.6f}')
                print(f'  ├─ L_pixel:  {avg_pixel:.6f} × {beta:.1f} = {weighted_pixel:.6f}')
                print(f'  └─ L_depth:  {avg_depth:.6f} × {lambda_depth:.1f} = {weighted_depth:.6f}')
                print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.2e}')
                
                rolling = {"loss": 0.0, "gradE": 0.0, "pixel": 0.0, "depth": 0.0}
                t0 = time.time()
        
        # Calculate epoch average losses
        num_batches = len(self.dataloader)
        epoch_avg_loss = epoch_totals["loss"] / num_batches if num_batches > 0 else 0.0
        epoch_avg_gradE = epoch_totals["gradE"] / num_batches if num_batches > 0 else 0.0
        epoch_avg_pixel = epoch_totals["pixel"] / num_batches if num_batches > 0 else 0.0
        epoch_avg_depth = epoch_totals["depth"] / num_batches if num_batches > 0 else 0.0
        
        # Record loss history
        self.loss_history['epoch'].append(epoch)
        self.loss_history['total'].append(epoch_avg_loss)
        self.loss_history['gradE'].append(epoch_avg_gradE)
        self.loss_history['pixel'].append(epoch_avg_pixel)
        self.loss_history['depth'].append(epoch_avg_depth)

        # Record depth quality history. Metrics are computed only when teacher depth is available.
        self.depth_quality_history['epoch'].append(epoch)
        if depth_quality_batches > 0:
            epoch_depth_mse = depth_quality_totals["mse"] / depth_quality_batches
            epoch_depth_mae = depth_quality_totals["mae"] / depth_quality_batches
            epoch_depth_rmse = depth_quality_totals["rmse"] / depth_quality_batches
            epoch_depth_psnr = depth_quality_totals["psnr"] / depth_quality_batches
            print(
                f"[Epoch {epoch}] Depth Quality: "
                f"MSE={epoch_depth_mse:.6f}, MAE={epoch_depth_mae:.6f}, "
                f"RMSE={epoch_depth_rmse:.6f}, PSNR={epoch_depth_psnr:.2f}dB"
            )
        else:
            epoch_depth_mse = float('nan')
            epoch_depth_mae = float('nan')
            epoch_depth_rmse = float('nan')
            epoch_depth_psnr = float('nan')
            print(f"[Epoch {epoch}] Depth Quality: teacher depth unavailable, metrics skipped")

        self.depth_quality_history['mse'].append(epoch_depth_mse)
        self.depth_quality_history['mae'].append(epoch_depth_mae)
        self.depth_quality_history['rmse'].append(epoch_depth_rmse)
        self.depth_quality_history['psnr'].append(epoch_depth_psnr)
        
        # Save checkpoint and plot curves
        self._save_checkpoint(epoch)
        self._save_training_history()
        self._plot_loss_curve()
        self._plot_depth_quality_curve()

    def _save_checkpoint(self, epoch: int) -> None:
        """Save full training checkpoint."""
        checkpoint_path = os.path.join(utils.MODEL_DIR, f"checkpoint_epoch_{epoch}.pth")
        latest_path = os.path.join(utils.MODEL_DIR, "latest.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "loss_history": self.loss_history,
            "depth_quality_history": self.depth_quality_history,
            "current_depth_dir": self.current_depth_dir,
            "sink_dims": utils.VAR_SINK_DIMS,
            "config": {
                "batch_size": utils.BATCH_SIZE,
                "gradient_accumulation_steps": utils.GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": utils.LEARNING_RATE,
                "n_epochs": utils.N_EPOCHS,
                "scheduler_start_epoch": utils.SCHEDULER_START_EPOCH,
                "scheduler_eta_min": getattr(utils, "SCHEDULER_ETA_MIN", 0.0),
                "base_ch": utils.BASE_CH,
                "use_depth": utils.USE_DEPTH,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, latest_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        print(f"Latest checkpoint updated: {latest_path}")

    def _resume_if_requested(self) -> None:
        """Resume training from a checkpoint when configured in utils.py."""
        resume_path = getattr(utils, "RESUME_CHECKPOINT", None)
        if not resume_path:
            return

        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"Resuming training from checkpoint: {resume_path}")
        loaded = torch.load(resume_path, map_location=self.device)

        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            self.model.load_state_dict(loaded["model_state_dict"])

            if "optimizer_state_dict" in loaded and loaded["optimizer_state_dict"] is not None:
                self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
            if "scheduler_state_dict" in loaded and loaded["scheduler_state_dict"] is not None:
                self.scheduler.load_state_dict(loaded["scheduler_state_dict"])
            if self.scaler is not None and loaded.get("scaler_state_dict") is not None:
                self.scaler.load_state_dict(loaded["scaler_state_dict"])

            self.loss_history = loaded.get("loss_history", self.loss_history)
            self.depth_quality_history = loaded.get("depth_quality_history", self.depth_quality_history)
            self.current_depth_dir = loaded.get("current_depth_dir", self.current_depth_dir)
            if loaded.get("sink_dims"):
                self._apply_sink_dims(torch.tensor(loaded["sink_dims"]))
            self.start_epoch = int(loaded.get("epoch", 0)) + 1
            print(f"Full training state restored. Continuing from epoch {self.start_epoch}.")
            return

        if isinstance(loaded, dict) and "state_dict" in loaded:
            self.model.load_state_dict(loaded["state_dict"])
            self.start_epoch = int(loaded.get("epoch", 0)) + 1
            print(
                "Loaded checkpoint with state_dict only. Optimizer/scheduler/history were not saved "
                f"in this file, so they start fresh. Continuing from epoch {self.start_epoch}."
            )
            return

        # Backward compatibility: old checkpoints saved only model.state_dict().
        self.model.load_state_dict(loaded)
        match = re.search(r"checkpoint_epoch_(\d+)\.pth$", os.path.basename(resume_path))
        if match:
            self.start_epoch = int(match.group(1)) + 1
        print(
            "Loaded legacy model-only checkpoint. Optimizer/scheduler/history were not saved "
            f"in this file, so they start fresh. Continuing from epoch {self.start_epoch}."
        )

    def _save_training_history(self) -> None:
        """Save the full epoch-level training history for the entire run."""
        history_path = os.path.join(utils.LOSS_DIR, 'training_history.csv')
        fieldnames = [
            'epoch',
            'total_loss',
            'gradE_loss',
            'pixel_loss',
            'depth_loss',
            'depth_mse',
            'depth_mae',
            'depth_rmse',
            'depth_psnr',
        ]

        with open(history_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, epoch in enumerate(self.loss_history['epoch']):
                writer.writerow({
                    'epoch': epoch,
                    'total_loss': self.loss_history['total'][i],
                    'gradE_loss': self.loss_history['gradE'][i],
                    'pixel_loss': self.loss_history['pixel'][i],
                    'depth_loss': self.loss_history['depth'][i],
                    'depth_mse': self.depth_quality_history['mse'][i],
                    'depth_mae': self.depth_quality_history['mae'][i],
                    'depth_rmse': self.depth_quality_history['rmse'][i],
                    'depth_psnr': self.depth_quality_history['psnr'][i],
                })

        print(f"Full training history saved: {history_path}")
    
    def _plot_loss_curve(self) -> None:
        """Plot and save loss curves."""
        if len(self.loss_history['epoch']) == 0:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Loss Convergence Curves', fontsize=16, fontweight='bold')
        epochs = self.loss_history['epoch']
        
        # Total loss
        axes[0, 0].plot(epochs, self.loss_history['total'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # L_gradE
        axes[0, 1].plot(epochs, self.loss_history['gradE'], 'r-', linewidth=2)
        axes[0, 1].set_title('L_gradE (Gradient Enhancement Loss)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # L_pixel
        axes[0, 2].plot(epochs, self.loss_history['pixel'], 'g-', linewidth=2)
        axes[0, 2].set_title('L_pixel (Intensity Loss)', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        # L_depth
        axes[1, 0].plot(epochs, self.loss_history['depth'], 'c-', linewidth=2)
        axes[1, 0].set_title('L_depth (Depth Loss)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # All losses together
        axes[1, 1].plot(epochs, self.loss_history['total'], 'b-', label='Total', linewidth=2)
        axes[1, 1].plot(epochs, self.loss_history['gradE'], 'r-', label='L_gradE', linewidth=1.5, alpha=0.7)
        axes[1, 1].plot(epochs, self.loss_history['pixel'], 'g-', label='L_pixel', linewidth=1.5, alpha=0.7)
        axes[1, 1].plot(epochs, self.loss_history['depth'], 'c-', label='L_depth', linewidth=1.5, alpha=0.7)
        axes[1, 1].set_title('All Losses', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        loss_curve_path = os.path.join(utils.LOSS_DIR, 'loss_curves.png')
        plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Loss curves saved: {loss_curve_path}")

    def _plot_depth_quality_curve(self) -> None:
        """Plot and save depth quality evolution curves."""
        if len(self.depth_quality_history['epoch']) == 0:
            return

        epochs = np.asarray(self.depth_quality_history['epoch'], dtype=np.float32)
        metrics = {
            'MSE': np.asarray(self.depth_quality_history['mse'], dtype=np.float32),
            'MAE': np.asarray(self.depth_quality_history['mae'], dtype=np.float32),
            'RMSE': np.asarray(self.depth_quality_history['rmse'], dtype=np.float32),
            'PSNR (dB)': np.asarray(self.depth_quality_history['psnr'], dtype=np.float32),
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Depth Map Quality Evolution Curves', fontsize=16, fontweight='bold')

        for ax, (title, values) in zip(axes.flat, metrics.items()):
            finite = np.isfinite(values)
            if finite.any():
                ax.plot(epochs[finite], values[finite], marker='o', linewidth=2)
            else:
                ax.text(
                    0.5, 0.5,
                    'No teacher depth metrics available',
                    ha='center', va='center', transform=ax.transAxes
                )
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        depth_curve_path = os.path.join(utils.LOSS_DIR, 'depth_quality_curves.png')
        plt.savefig(depth_curve_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Depth quality curves saved: {depth_curve_path}")

    def train(self) -> None:
        """Main training loop."""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")
        
        # Stage 1: independently pre-train F for 70 epochs while D is frozen.
        for epoch in range(self.start_epoch, utils.PRETRAIN_EPOCHS + 1):
            self._freeze_depth_branch()
            self.dataloader = self._setup_dataloader(epoch)
            self.train_epoch(epoch, "Stage 1: pre-train F (D frozen)", use_depth_loss=False)

        # Calibrate D_sink and generate the initial teacher from the pre-trained F.
        if self.current_depth_dir is None and self.start_epoch <= utils.PRETRAIN_EPOCHS + 1:
            self._calibrate_sink_channels()
            self._update_depth_maps(utils.PRETRAIN_EPOCHS)

        # Stage 2: seven cycles of 5 frozen-D and 5 joint epochs.
        for cycle in range(1, utils.NUM_SYNERGY_CYCLES + 1):
            start = utils.PRETRAIN_EPOCHS + (cycle - 1) * (utils.FUSION_EPOCHS_PER_CYCLE + utils.JOINT_EPOCHS_PER_CYCLE) + 1
            end = start + utils.FUSION_EPOCHS_PER_CYCLE + utils.JOINT_EPOCHS_PER_CYCLE - 1
            if self.start_epoch > end:
                continue
            print(f"\n{'='*80}\nStage 2 — cycle {cycle}/{utils.NUM_SYNERGY_CYCLES}\n{'='*80}")
            self._freeze_depth_branch()
            for epoch in range(start, start + utils.FUSION_EPOCHS_PER_CYCLE):
                if epoch >= self.start_epoch:
                    self._set_dataloader_for_split(epoch, "train1")
                    self.train_epoch(epoch, "Step 1: Dtrain1, optimize F", use_depth_loss=False)
                    self.scheduler.step()
            self._unfreeze_depth_branch()
            for epoch in range(start + utils.FUSION_EPOCHS_PER_CYCLE, end + 1):
                if epoch >= self.start_epoch:
                    self._set_dataloader_for_split(epoch, "train2")
                    self.train_epoch(epoch, "Step 2: Dtrain2, optimize F+D", use_depth_loss=True)
                    self.scheduler.step()
            self._update_depth_maps(end)
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
