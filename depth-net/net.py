# ===================================================================================
# net.py - 主干网络定义文件
# ===================================================================================
# 本文件包含以下内容：
# 1. 基础模块（Basic Modules）：DoubleConv, Down, EnhancedDown, Up等
# 2. FNet2D模块（FNet2D Modules）：FourierMix2D, FFN, FNet2DBlock
# 3. 主干网络（Main Networks）：DualBranchUNet, Net
#
# 从moudle.py导入的模块：
# - PaperHDGFFM: 融合模块（用于VIS-深度融合和IR-VIS融合）
# - DepthBranchF: 深度分支（用于深度估计和特征提取）
# ===================================================================================

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torch.cuda.amp import custom_fwd
from moudle import PaperHDGFFM, DepthBranchF
import utils


def norm_1(x):
    """Normalize tensor to [0, 1] range."""
    max1 = torch.max(x)
    min1 = torch.min(x)
    norm = (x - min1) / (max1 - min1 + 1e-10)
    return norm


# =========================
# LayerNorm2d for 2D feature maps
# =========================
class LayerNorm2d(nn.Module):
    """
    LayerNorm for 2D feature maps: [B, C, H, W]
    Normalizes over (C, H, W) for each sample independently.
    Compatible with nn.BatchNorm2d interface.
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1]
        var  = x.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def get_norm_layer(num_channels):
    """
    根据 batch size 选择归一化层
    - 小 batch (<=4): LayerNorm2d（稳定，不依赖batch统计）
    - 大 batch (>4): BatchNorm2d（快速，利用batch统计）
    
    Args:
        num_channels: 通道数
    
    Returns:
        nn.Module: 归一化层（LayerNorm2d 或 BatchNorm2d）
    """
    if utils.BATCH_SIZE <= 4:
        return LayerNorm2d(num_channels)
    else:
        return nn.BatchNorm2d(num_channels)


# =========================
# FNet2D Blocks for Global Feature Extraction
# =========================
class FourierMix2D(nn.Module):
    """2D Fourier Mixing Module: FFT-based global feature mixing."""
    def __init__(self, channels, use_imag=True):
        super().__init__()
        self.use_imag = use_imag
        # 频域门控（可学习的缩放），提升稳定性
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        # 若拼接实/虚部，回投一层
        self.proj = nn.Conv2d(channels * (2 if use_imag else 1), channels, 1, bias=True)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        B, C, H, W = x.shape
        # 2D FFT：实输入 → 复输出，形状 [B,C,H,W/2+1]
        # 强制使用 float32 避免 float16 下的数值问题
        Xf = torch.fft.rfft2(x, norm="ortho")
        # 幅度归一化（可选，提升数值稳定）
        scale = (H * W) ** 0.5
        Xf = Xf / scale

        if self.use_imag:
            feat = torch.cat([Xf.real, Xf.imag], dim=1)  # [B, 2C, H, W/2+1]
        else:
            feat = Xf.real  # [B, C, H, W/2+1]

        # 线性回投到 C 通道（频域上做一次可学习"混合"）
        feat = self.proj(feat)

        # 恢复成复数，简洁起见把虚部从0起步（也可学一个虚部分支）
        Xf_new = torch.complex(feat, torch.zeros_like(feat))

        # 逆 FFT 回到空间域
        y = torch.fft.irfft2(Xf_new * scale, s=(H, W), norm="ortho")
        # 残差与门控
        return self.alpha * y + self.beta * x


class FFN(nn.Module):
    """Feed-Forward Network: Pointwise → Depthwise → Pointwise."""
    def __init__(self, dim, expansion=2.66, dw_kernel=3):
        super().__init__()
        hidden = int(dim * expansion)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, dw_kernel, padding=dw_kernel//2, groups=hidden)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        r = x
        x = self.pw1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.pw2(x)
        return x + r


class FNet2DBlock(nn.Module):
    """FNet2D Block: FourierMix + FFN with residual connections."""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = get_norm_layer(dim)
        self.mix = FourierMix2D(dim, use_imag=True)
        self.norm2 = get_norm_layer(dim)
        self.ffn = FFN(dim)

    def forward(self, x):
        x = x + self.mix(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DoubleConv(nn.Module):
    """Double convolution block: Conv-Norm-ReLU-Conv-Norm-ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            get_norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            get_norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EnhancedDown(nn.Module):
    """
    Enhanced Downsampling with Split → Two Branches → Concat design.
    
    Architecture:
        Input → Conv2d(stride=2) → Split (channel dimension) 
        → Branch1: DoubleConv (local features)
        → Branch2: FNet2DBlock (global features)
        → Concat → Output
    
    This design combines local and global features efficiently.
    Uses Conv2d(stride=2) instead of MaxPool2d to match depth branch downsampling.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Ensure channels are even for splitting
        assert in_ch % 2 == 0, f"in_ch ({in_ch}) must be even for splitting"
        assert out_ch % 2 == 0, f"out_ch ({out_ch}) must be even for splitting"
        
        # Use Conv2d(stride=2) instead of MaxPool2d to match depth branch
        # This ensures consistent spatial dimensions with depth branch
        self.downsample = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=False)
        
        # Branch 1: Local feature extraction (DoubleConv)
        # Processes first half of channels: in_ch//2 → out_ch//2
        self.branch1 = DoubleConv(in_ch // 2, out_ch // 2)
        
        # Branch 2: Global feature extraction (FNet2DBlock)
        # Processes second half of channels: in_ch//2 → out_ch//2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch // 2, out_ch // 2, 1, bias=False),  # Channel adjustment
            get_norm_layer(out_ch // 2),
            FNet2DBlock(out_ch // 2)  # Global feature extraction
        )
        
        # Optional: Post-processing after concatenation
        self.post_norm = get_norm_layer(out_ch)

    def forward(self, x):
        # Downsampling using Conv2d(stride=2) to match depth branch
        # Output size: floor((H-1)/2 + 1) × floor((W-1)/2 + 1)
        downsampled = self.downsample(x)  # [B, in_ch, H', W']
        
        # Split channels into two halves
        ch1, ch2 = downsampled.chunk(2, dim=1)  # Each: [B, in_ch//2, H', W']
        
        # Branch 1: Local features (DoubleConv)
        branch1_out = self.branch1(ch1)  # [B, out_ch//2, H', W']
        
        # Branch 2: Global features (FNet2DBlock)
        branch2_out = self.branch2(ch2)  # [B, out_ch//2, H', W']
        
        # Concatenate features
        out = torch.cat([branch1_out, branch2_out], dim=1)  # [B, out_ch, H', W']
        
        # Post-processing: normalization
        out = self.post_norm(out)
        
        return out


class Up(nn.Module):
    """Upsampling: Bilinear Interpolation + Conv + Concat + DoubleConv"""
    def __init__(self, in_ch_up, in_ch_skip, out_ch):
        super().__init__()
        # Use bilinear interpolation instead of ConvTranspose2d
        # 1x1 conv to adjust channels after upsampling
        self.up_conv = nn.Conv2d(in_ch_up, out_ch, 1)
        self.conv = DoubleConv(in_ch_skip + out_ch, out_ch)

    def forward(self, x_up, x_skip):
        # Bilinear interpolation to match skip connection size
        x = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)
        # Adjust channels
        x = self.up_conv(x)
        # Concatenate with skip connection
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)


# ===================================================================================
# 深度分支模块已移动到 moudle.py
# - ConvGNAct, DepthDown, DepthUp, DepthBranchF 都在 moudle.py 中定义
# - 从 moudle 导入 DepthBranchF
# ===================================================================================


class DualBranchUNet(nn.Module):
    """
    双分支U-Net，用于IR-VIS图像融合，支持深度分支
    
    功能：
        - IR分支编码器：提取红外图像特征
        - VIS分支编码器：提取可见光图像特征（可选融合深度特征）
        - 深度分支：深度估计和特征提取（可选）
        - 瓶颈融合：使用PaperHDGFFM融合IR和VIS特征
        - 解码器：使用跳跃连接重建融合图像
    
    架构：
        - 编码器：5层下采样（Layer1使用DoubleConv，Layer2-5使用EnhancedDown）
        - 深度分支：5层编码器，金字塔结构，多尺度深度预测
        - VIS-深度融合：在vis2~vis5层使用PaperHDGFFM融合深度特征
        - 瓶颈融合：在vis5和ir5层使用PaperHDGFFM融合
        - 解码器：4层上采样，使用IR和VIS的跳跃连接
    
    使用位置：
        - Net类中使用（作为主要网络）
        - 训练和测试时使用
    
    Args:
        base_ch: 基础通道数（默认从utils.py读取）
        use_depth: 是否使用深度分支（默认从utils.py读取）
    """
    
    def __init__(self, base_ch=None, use_depth=None):
        super().__init__()
        # Use values from utils.py if not provided
        if base_ch is None:
            base_ch = utils.BASE_CH
        # Depth branch always exists
        self.use_depth = True
        
        # IR branch encoder
        self.ir_inc = DoubleConv(1, base_ch)  # Layer1: 1→16 (保持不变)
        # Layer2-5: Use EnhancedDown with Split → Two Branches → Concat design
        self.ir_down1 = EnhancedDown(base_ch, base_ch * 2)  # 16→32
        self.ir_down2 = EnhancedDown(base_ch * 2, base_ch * 4)  # 32→64
        self.ir_down3 = EnhancedDown(base_ch * 4, base_ch * 8)  # 64→128
        self.ir_down4 = EnhancedDown(base_ch * 8, base_ch * 16)  # 128→256
        
        # VIS branch encoder
        self.vis_inc = DoubleConv(1, base_ch)  # Layer1: 1→16 (保持不变)
        # Layer2-5: Use EnhancedDown with Split → Two Branches → Concat design
        self.vis_down1 = EnhancedDown(base_ch, base_ch * 2)  # 16→32
        self.vis_down2 = EnhancedDown(base_ch * 2, base_ch * 4)  # 32→64
        self.vis_down3 = EnhancedDown(base_ch * 4, base_ch * 8)  # 64→128
        self.vis_down4 = EnhancedDown(base_ch * 8, base_ch * 16)  # 128→256
        
        # Depth branch F (always exists)
        chs = (base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16)
        self.depth_branch = DepthBranchF(
            in_ch=utils.DEPTH_BRANCH_IN_CH,
            chs=chs,
            regress_depth=utils.DEPTH_BRANCH_REGRESS_DEPTH
        )
        # Only create fusion modules for vis2~vis5 (skip vis1)
        # Channels: [32, 64, 128, 256] corresponding to vis2, vis3, vis4, vis5
        self.vis_depth_fusions = nn.ModuleList()
        for idx, ch in enumerate(chs[1:]):  # vis2~vis5
            use_var = utils.VAR_ENABLED and utils.VAR_ENABLE_VIS_DEPTH and (idx == len(chs[1:]) - 1)
            self.vis_depth_fusions.append(
                PaperHDGFFM(
                    c_img=ch, 
                    c_dep=ch, 
                    heads=utils.FUSION_HEADS, 
                    tau=utils.FUSION_TAU, 
                    with_norm=utils.FUSION_WITH_NORM,
                    enable_var=use_var
                )
            )
        
        # Fusion at bottleneck using PaperHDGFFM
        # A=vis5 (VIS特征，已融合深度), B=ir5 (IR特征)
        # 内部会先对VIS做自注意力，再做VIS←IR交叉注意力
        self.bottleneck_fusion = PaperHDGFFM(
            c_img=base_ch * 16,      # 256 (VIS特征通道)
            c_dep=base_ch * 16,      # 256 (IR特征通道)
            heads=utils.FUSION_HEADS,
            tau=utils.FUSION_TAU,
            with_norm=utils.FUSION_WITH_NORM,
            enable_var=(utils.VAR_ENABLED and utils.VAR_ENABLE_VIS_IR)
        )
        
        # Decoder (input channels doubled due to concatenated skip connections)
        self.up1 = Up(base_ch * 16, base_ch * 8 * 2, base_ch * 8)  # skip: ir4+vis4
        self.up2 = Up(base_ch * 8, base_ch * 4 * 2, base_ch * 4)   # skip: ir3+vis3
        self.up3 = Up(base_ch * 4, base_ch * 2 * 2, base_ch * 2)    # skip: ir2+vis2
        self.up4 = Up(base_ch * 2, base_ch * 2, base_ch)            # skip: ir1+vis1
        self.outc = nn.Conv2d(base_ch, 1, 1)

    def forward(self, ir, vis):
        # IR branch encoding
        ir1 = self.ir_inc(ir)
        ir2 = self.ir_down1(ir1)
        ir3 = self.ir_down2(ir2)
        ir4 = self.ir_down3(ir3)
        ir5 = self.ir_down4(ir4)

        # VIS branch encoding with progressive depth fusion
        # Extract dual-stream depth features progressively from VIS and IR
        dep_feats = self.depth_branch.forward_encoder(vis, ir)
        dep1, dep2, dep3, dep4, dep5 = dep_feats

        # VIS encoding: vis1 without depth fusion, vis2~vis5 with HDGFFM fusion
        # vis1: No depth fusion (skip depth injection)
        vis1 = self.vis_inc(vis)

        # vis2: HDGFFM fusion with dep2
        vis2_raw = self.vis_down1(vis1)
        vis2, _ = self.vis_depth_fusions[0](vis2_raw, dep2)

        # vis3: HDGFFM fusion with dep3
        vis3_raw = self.vis_down2(vis2)
        vis3, _ = self.vis_depth_fusions[1](vis3_raw, dep3)

        # vis4: HDGFFM fusion with dep4
        vis4_raw = self.vis_down3(vis3)
        vis4, _ = self.vis_depth_fusions[2](vis4_raw, dep4)

        # vis5: HDGFFM fusion with dep5
        vis5_raw = self.vis_down4(vis4)
        vis5, _ = self.vis_depth_fusions[3](vis5_raw, dep5)

        # Depth regression (pyramid structure handles this internally)
        d_hat = None
        if self.depth_branch.regress_depth:
            # Call depth_branch.forward() to get d_hat using pyramid structure
            # Note: We already have dep_feats from forward_encoder(),
            # but forward() will recompute them (slight redundancy but cleaner code)
            d_hat, _ = self.depth_branch(vis, ir)
        
        # Fusion at bottleneck using PaperHDGFFM
        # 交换输入顺序：vis5作为A（会先做自注意力），ir5作为B
        fused, _ = self.bottleneck_fusion(vis5, ir5)
        
        # Decoding with skip connections from both branches (concatenate IR and VIS features)
        d4 = self.up1(fused, torch.cat([ir4, vis4], dim=1))  # Fuse IR and VIS skip connections
        d3 = self.up2(d4, torch.cat([ir3, vis3], dim=1))    # Fuse IR and VIS skip connections
        d2 = self.up3(d3, torch.cat([ir2, vis2], dim=1))     # Fuse IR and VIS skip connections
        d1 = self.up4(d2, torch.cat([ir1, vis1], dim=1))     # Fuse IR and VIS skip connections
        
        out = self.outc(d1)
        return out, d_hat


class Net(nn.Module):
    """
    主融合网络：双分支U-Net + 深度分支
    
    功能：
        - 封装DualBranchUNet网络
        - 处理输入输出归一化
        - 训练和测试模式的不同输出
    
    使用位置：
        - train.py (训练时使用)
        - test.py (测试时使用)
    
    Args:
        use_depth: 是否使用深度分支（默认从utils.py读取）
        base_ch: 基础通道数（默认从utils.py读取）
    """
    
    def __init__(self, use_depth=None, base_ch=None):
        super(Net, self).__init__()
        # Use values from utils.py if not provided
        # Depth branch always exists
        if base_ch is None:
            base_ch = utils.BASE_CH
        self.dual_unet = DualBranchUNet(base_ch=base_ch, use_depth=True)

    def forward(self, IR, VIS):
        """
        Forward pass for image fusion.
        
        Args:
            IR: Infrared image [B, 1, H, W]
            VIS: Visible image [B, 1, H, W]
        
        Returns:
            fused: Fused image [B, 1, H, W]
            Y: Fusion component [B, 1, H, W]
            d_hat: Predicted depth [B, 1, H, W] (None if regress_depth=False)
        """
        # Dual-branch U-Net fusion with depth branch
        Y_final, d_hat = self.dual_unet(IR, VIS)
        Y_final = norm_1(Y_final)

        if self.training:
            # Return zero loss for training compatibility
            zero_loss = torch.zeros(1, device=Y_final.device, dtype=Y_final.dtype)
            return (VIS + Y_final), zero_loss, d_hat
        else:
            return (VIS + Y_final), Y_final, d_hat
