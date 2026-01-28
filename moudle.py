# ===================================================================================
# moudle.py - 模块集合文件
# ===================================================================================
# 本文件包含以下模块：
# 1. 数据集模块（Dataset Module）：图像加载和数据集管理
# 2. 深度分支模块（Depth Branch Module）：深度估计网络相关组件
# 3. 融合模块（Fusion Module）：基于相关性交叉注意力的融合模块
#
# 使用位置：
# - 数据集模块：train.py (训练时加载数据)
# - 深度分支模块：net.py (DepthBranchF用于深度估计和特征提取)
# - 融合模块：net.py (PaperHDGFFM用于VIS和深度特征的融合)
# ===================================================================================

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import glob
import numpy as np
import torch.utils.data as data1
from PIL import Image
import utils


# ===================================================================================
# 第一部分：数据集模块 (Dataset Module)
# ===================================================================================
# 功能：图像加载和数据集管理
# 使用位置：train.py (训练时使用fusiondata类加载数据)
# ===================================================================================

def load_image(x):
    """
    加载灰度图像并转换为tensor
    
    功能：
        - 从文件路径加载图像
        - 转换为灰度图
        - 转换为torch.Tensor格式
    
    使用位置：
        - fusiondata.__getitem__() 内部调用
        - 用于加载IR、VIS和深度图像
    
    Args:
        x: 图像文件路径
    
    Returns:
        tensor: [1, H, W] 格式的tensor
    """
    try:
        img = Image.open(x)
        img = img.convert('L')
        arr = np.asarray(img, dtype=np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor
    except Exception as e:
        raise RuntimeError(f"Failed to read image: {x}. Error: {e}")


def _list_images_sorted(folder):
    """
    列出并排序文件夹中的图像文件，确保唯一的基础名称
    
    功能：
        - 扫描文件夹中的所有图像文件
        - 按文件名中的数字排序
        - 处理重复文件（按扩展名优先级选择）
    
    使用位置：
        - make_dataset() 内部调用
        - 用于获取IR、VIS和深度图像列表
    
    Args:
        folder: 图像文件夹路径
    
    Returns:
        files: 排序后的图像文件路径列表
    """
    patterns = ["*.jpg", "*.png", "*.bmp", "*.jpeg", "*.tiff", "*.tif", 
                "*.JPG", "*.PNG", "*.BMP", "*.JPEG", "*.TIFF", "*.TIF"]
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(os.path.join(folder, p)))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No images found in '{folder}'")
    
    # Group files by basename (without extension) to handle duplicates
    # Priority: .bmp > .png > .jpg > .jpeg > .tiff > .tif
    ext_priority = {'.bmp': 0, '.BMP': 0, '.png': 1, '.PNG': 1, 
                    '.jpg': 2, '.JPG': 2, '.jpeg': 3, '.JPEG': 3,
                    '.tiff': 4, '.TIFF': 4, '.tif': 5, '.TIF': 5}
    
    file_dict = {}
    for f in all_files:
        basename = os.path.splitext(os.path.basename(f))[0]
        ext = os.path.splitext(f)[1]
        if basename not in file_dict:
            file_dict[basename] = f
        else:
            # Keep file with higher priority extension
            current_ext = os.path.splitext(file_dict[basename])[1]
            if ext_priority.get(ext, 99) < ext_priority.get(current_ext, 99):
                file_dict[basename] = f
    
    files = list(file_dict.values())
    
    def _num(name):
        """提取文件名中的数字用于排序，支持纯数字命名（如 1.bmp, 2.bmp, 3.bmp）"""
        base = os.path.basename(name)
        # 提取基础名称（去除扩展名）
        basename = os.path.splitext(base)[0]
        # 如果基础名称本身就是纯数字（如 "1", "2", "3"），直接转换
        if basename.isdigit():
            return int(basename)
        # 否则尝试从文件名中提取第一个数字
        m = re.search(r"(\d+)", basename)
        return int(m.group(1)) if m else 0
    
    files.sort(key=_num)
    return files


def make_dataset(ir_dir, vi_dir, depth_dir=None):
    """
    从IR和VIS目录创建数据集对，按基础名称匹配
    
    功能：
        - 匹配IR和VIS图像对（按文件名去除扩展名）
        - 可选匹配深度图像
        - 返回匹配的图像对列表
    
    使用位置：
        - fusiondata.__init__() 内部调用
        - 用于创建训练数据集
    
    Args:
        ir_dir: IR图像目录
        vi_dir: VIS图像目录
        depth_dir: 可选的深度图像目录（用于teacher depth）
    
    Returns:
        pairs: 图像对列表，格式为 (ir_path, vi_path, depth_path) 或 (ir_path, vi_path)
    """
    ir_list = _list_images_sorted(ir_dir)
    vi_list = _list_images_sorted(vi_dir)
    
    # Create dictionaries mapping basename to full path
    ir_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in ir_list}
    vi_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in vi_list}
    
    # Find common basenames
    # 支持纯数字命名排序（如 "1", "2", "3", "10" 会正确排序为 1, 2, 3, 10）
    def _sort_key(basename):
        """排序函数，支持纯数字命名（如 '1', '2', '3'）"""
        # 如果基础名称本身就是纯数字，直接转换
        if basename.isdigit():
            return int(basename)
        # 否则尝试提取数字
        m = re.search(r'(\d+)', basename)
        return int(m.group(1)) if m else 0
    
    common_basenames = sorted(set(ir_dict.keys()) & set(vi_dict.keys()), key=_sort_key)
    
    # Create pairs
    if depth_dir is not None:
        depth_list = _list_images_sorted(depth_dir)
        depth_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in depth_list}
        # Only keep pairs where depth also exists
        common_basenames = [bn for bn in common_basenames if bn in depth_dict]
        pairs = [(ir_dict[bn], vi_dict[bn], depth_dict[bn]) for bn in common_basenames]
    else:
        pairs = [(ir_dict[bn], vi_dict[bn]) for bn in common_basenames]
    
    return pairs


class fusiondata(data1.Dataset):
    """
    图像融合训练数据集类，支持可选的深度监督
    
    功能：
        - 加载IR、VIS和深度图像对
        - 支持有深度和无深度两种情况
        - 用于PyTorch DataLoader
    
    使用位置：
        - train.py (Trainer类中创建数据集)
        - 训练时通过DataLoader加载数据
    
    Args:
        ir_dir: IR图像目录
        vi_dir: VIS图像目录
        depth_dir: 可选的深度图像目录
        train: 是否为训练模式
    """
    def __init__(self, ir_dir, vi_dir, depth_dir=None, train=True):
        self.train = train
        self.has_depth = depth_dir is not None
        if self.train:
            self.train_set_path = make_dataset(ir_dir, vi_dir, depth_dir)

    def __getitem__(self, idx):
        if self.train:
            if self.has_depth:
                imgA_path, imgB_path, depth_path = self.train_set_path[idx]
                imgA = load_image(imgA_path)
                imgB = load_image(imgB_path)
                depth = load_image(depth_path)
                return imgA, imgB, depth
            else:
                imgA_path, imgB_path = self.train_set_path[idx]
                imgA = load_image(imgA_path)
                imgB = load_image(imgB_path)
                return imgA, imgB

    def __len__(self):
        if self.train:
            return len(self.train_set_path)


# ===================================================================================
# 第二部分：深度分支模块 (Depth Branch Module)
# ===================================================================================
# 功能：深度估计网络相关组件
# 使用位置：net.py (DepthBranchF用于深度估计和特征提取)
# ===================================================================================

class ConvGNAct(nn.Module):
    """
    卷积 + 组归一化 + GELU激活
    
    功能：
        - 深度分支的基础卷积块
        - 使用GroupNorm代替BatchNorm（更适合小batch）
        - GELU激活函数
    
    使用位置：
        - DepthBranchF的编码器和解码器中
        - 用于深度特征提取
    
    Args:
        cin: 输入通道数
        cout: 输出通道数
        k: 卷积核大小
        s: 步长
        p: 填充（None时自动计算）
        groups: GroupNorm的组数
    """
    def __init__(self, cin, cout, k=3, s=1, p=None, groups=8):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, s, p)
        self.gn = nn.GroupNorm(min(groups, cout), cout)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class DepthDown(nn.Module):
    """
    深度分支的下采样模块
    
    功能：
        - 使用Conv2d(stride=2)进行下采样
        - 保持通道数不变
        - 与VIS/IR分支的下采样方式一致（确保空间尺寸匹配）
    
    使用位置：
        - DepthBranchF的编码器中
        - 用于逐步降低特征图分辨率
    
    Args:
        c: 通道数
    """
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class DepthUp(nn.Module):
    """
    深度分支的上采样模块
    
    功能：
        - 使用双线性插值进行上采样
        - 1x1卷积调整通道数
        - 用于金字塔结构的上采样路径
    
    使用位置：
        - DepthBranchF的解码器中（金字塔结构）
        - 用于多尺度深度预测的上采样
    
    Args:
        cin: 输入通道数
        cout: 输出通道数
    """
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proj = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.proj(self.up(x))


class DepthBranchF(nn.Module):
    """
    深度分支F，具有金字塔结构（FPN风格）
    
    功能：
        - 5层编码器，与VIS分支对齐
        - 自上而下的路径，带有横向连接（金字塔结构）
        - 多尺度深度预测和融合
        - 提供编码器特征用于VIS-深度融合
    
    使用位置：
        - net.py (DualBranchUNet中使用)
        - 用于深度估计和深度特征提取
        - 提取的特征用于PaperHDGFFM融合模块
    
    架构：
        - 编码器（自下而上）：5层卷积，逐步下采样
        - 金字塔结构（自上而下）：横向连接 + 上采样
        - 多尺度深度预测：在5个尺度上预测深度，然后融合
    
    Args:
        in_ch: 输入通道数（默认1，灰度图）
        chs: 各层通道数 (默认: 16, 32, 64, 128, 256)
        regress_depth: 是否回归深度预测（True时进行深度预测，False时只提取特征）
    """
    def __init__(self, in_ch=1, chs=(16, 32, 64, 128, 256), regress_depth=True):
        super().__init__()
        c1, c2, c3, c4, c5 = chs
        
        # Encoder (5 layers) - bottom-up pathway
        self.enc1 = nn.Sequential(ConvGNAct(in_ch, c1), ConvGNAct(c1, c1))
        self.down1 = DepthDown(c1)
        self.enc2 = nn.Sequential(ConvGNAct(c1, c2), ConvGNAct(c2, c2))
        self.down2 = DepthDown(c2)
        self.enc3 = nn.Sequential(ConvGNAct(c2, c3), ConvGNAct(c3, c3))
        self.down3 = DepthDown(c3)
        self.enc4 = nn.Sequential(ConvGNAct(c3, c4), ConvGNAct(c4, c4))
        self.down4 = DepthDown(c4)
        self.enc5 = nn.Sequential(ConvGNAct(c4, c5), ConvGNAct(c5, c5))
        
        self.regress_depth = regress_depth
        if regress_depth:
            # Lateral connections (1x1 conv to reduce channels for pyramid fusion)
            # These align encoder features for pyramid fusion
            self.lateral5 = nn.Conv2d(c5, c5, 1)
            self.lateral4 = nn.Conv2d(c4, c4, 1)
            self.lateral3 = nn.Conv2d(c3, c3, 1)
            self.lateral2 = nn.Conv2d(c2, c2, 1)
            self.lateral1 = nn.Conv2d(c1, c1, 1)
            
            # Top-down pathway (pyramid fusion)
            # Each level fuses top-down feature with lateral connection
            self.topdown5 = nn.Sequential(ConvGNAct(c5, c5), ConvGNAct(c5, c5))
            self.topdown4 = nn.Sequential(ConvGNAct(c4 + c5, c4), ConvGNAct(c4, c4))
            self.topdown3 = nn.Sequential(ConvGNAct(c3 + c4, c3), ConvGNAct(c3, c3))
            self.topdown2 = nn.Sequential(ConvGNAct(c2 + c3, c2), ConvGNAct(c2, c2))
            self.topdown1 = nn.Sequential(ConvGNAct(c1 + c2, c1), ConvGNAct(c1, c1))
            
            # Multi-scale depth prediction heads
            # Predict depth at different scales, then fuse
            self.head5 = nn.Conv2d(c5, 1, 1)  # H/16 scale
            self.head4 = nn.Conv2d(c4, 1, 1)  # H/8 scale
            self.head3 = nn.Conv2d(c3, 1, 1)  # H/4 scale
            self.head2 = nn.Conv2d(c2, 1, 1)  # H/2 scale
            self.head1 = nn.Conv2d(c1, 1, 1)  # H scale
            
            # Final fusion layer (weighted combination of multi-scale predictions)
            self.fusion_conv = nn.Sequential(
                ConvGNAct(5, 16, 3, 1, 1),  # 5 scales → 16 channels
                ConvGNAct(16, 8, 3, 1, 1),
                nn.Conv2d(8, 1, 1)  # Final depth prediction
            )
    
    def forward_encoder(self, x) -> List[torch.Tensor]:
        """
        只前向传播编码器部分，返回每个阶段的特征
        
        功能：
            - 提取编码器各层特征
            - 用于VIS-深度融合（PaperHDGFFM）
        
        使用位置：
            - net.py (DualBranchUNet.forward()中调用)
            - 提取深度特征用于融合
        
        Args:
            x: 输入图像 [B, 1, H, W]
        
        Returns:
            dep_feats: 各层特征列表 [f1, f2, f3, f4, f5]
        """
        f1 = self.enc1(x)
        x = self.down1(f1)
        f2 = self.enc2(x)
        x = self.down2(f2)
        f3 = self.enc3(x)
        x = self.down3(f3)
        f4 = self.enc4(x)
        x = self.down4(f4)
        f5 = self.enc5(x)
        return [f1, f2, f3, f4, f5]
    
    def forward(self, x) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """
        完整前向传播：编码器 + 金字塔结构 + 深度预测
        
        功能：
            - 如果regress_depth=True，进行深度预测
            - 如果regress_depth=False，只返回编码器特征
        
        使用位置：
            - net.py (DualBranchUNet.forward()中调用，当需要深度预测时)
        
        Args:
            x: 输入图像 [B, 1, H, W]
        
        Returns:
            d_hat: 深度预测 [B, 1, H, W] (如果regress_depth=False则为None)
            dep_feats: 编码器特征列表 [f1, f2, f3, f4, f5]
        """
        # Bottom-up pathway (encoder)
        f1 = self.enc1(x)
        x = self.down1(f1)
        f2 = self.enc2(x)
        x = self.down2(f2)
        f3 = self.enc3(x)
        x = self.down3(f3)
        f4 = self.enc4(x)
        x = self.down4(f4)
        f5 = self.enc5(x)
        
        dep_feats = [f1, f2, f3, f4, f5]
        
        if not self.regress_depth:
            return None, dep_feats
        
        # Top-down pathway with lateral connections (Pyramid Structure)
        # Level 5 (highest level, smallest resolution)
        p5 = self.lateral5(f5)
        p5 = self.topdown5(p5)
        d5 = self.head5(p5)  # [B, 1, H/16, W/16]
        
        # Level 4: upsample p5 and fuse with lateral f4
        p5_up = F.interpolate(p5, size=f4.shape[2:], mode='bilinear', align_corners=False)
        p4 = self.lateral4(f4)
        p4 = torch.cat([p4, p5_up], dim=1)
        p4 = self.topdown4(p4)
        d4 = self.head4(p4)  # [B, 1, H/8, W/8]
        
        # Level 3: upsample p4 and fuse with lateral f3
        p4_up = F.interpolate(p4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.lateral3(f3)
        p3 = torch.cat([p3, p4_up], dim=1)
        p3 = self.topdown3(p3)
        d3 = self.head3(p3)  # [B, 1, H/4, W/4]
        
        # Level 2: upsample p3 and fuse with lateral f2
        p3_up = F.interpolate(p3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lateral2(f2)
        p2 = torch.cat([p2, p3_up], dim=1)
        p2 = self.topdown2(p2)
        d2 = self.head2(p2)  # [B, 1, H/2, W/2]
        
        # Level 1 (finest level, full resolution)
        p2_up = F.interpolate(p2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.lateral1(f1)
        p1 = torch.cat([p1, p2_up], dim=1)
        p1 = self.topdown1(p1)
        d1 = self.head1(p1)  # [B, 1, H, W]
        
        # Multi-scale depth fusion: upsample all predictions to full resolution and fuse
        d5_full = F.interpolate(d5, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d4_full = F.interpolate(d4, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d3_full = F.interpolate(d3, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d2_full = F.interpolate(d2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate all multi-scale predictions
        multi_scale_depths = torch.cat([d1, d2_full, d3_full, d4_full, d5_full], dim=1)  # [B, 5, H, W]
        
        # Final fusion to get single depth prediction
        d_hat = self.fusion_conv(multi_scale_depths)  # [B, 1, H, W]
        
        return d_hat, dep_feats


# ===================================================================================
# 第三部分：融合模块 (Fusion Module)
# ===================================================================================
# 功能：基于相关性交叉注意力的融合模块
# 使用位置：net.py (PaperHDGFFM用于VIS和深度特征的融合，以及IR和VIS特征的融合)
# ===================================================================================

# ------------------ Utils ------------------
@torch.jit.script
def _cosine_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    计算两个特征图的余弦相关性
    
    功能：
        - 计算通道维度的余弦相似度
        - 用于相关性门控融合
    
    使用位置：
        - PaperCorrCrossAttn.forward() 内部调用
        - 用于计算A_sa和A_ca的相关性
    
    Args:
        a: 特征图A [B, C, H, W]
        b: 特征图B [B, C, H, W]
        eps: 防止除零的小值
    
    Returns:
        corr: 余弦相关性 [B, 1, H, W]
    """
    # a,b: [B,C,H,W] -> [B,1,H,W]
    num = (a * b).sum(dim=1, keepdim=True)
    den = a.norm(p=2, dim=1, keepdim=True) * b.norm(p=2, dim=1, keepdim=True) + eps
    return num / den


class DepthwiseFFN(nn.Module):
    """
    深度可分离前馈网络：Eq.(10) 中的 H(·)
    
    功能：
        - 深度可分离卷积 + GELU + 1x1卷积
        - 保持通道数不变
        - 轻量级特征变换
    
    使用位置：
        - PaperCorrCrossAttn.forward() 内部调用
        - 用于特征融合后的非线性变换
    
    Args:
        c: 通道数
    """
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c),
            nn.GELU(),
            nn.Conv2d(c, c, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------ Core: Paper Corr-based Cross-Attn (Eq.5~10) ------------------
class PaperCorrCrossAttn(nn.Module):
    """
    基于相关性交叉注意力的融合模块（单尺度）
    
    功能：
        - Eq.(5) A的自注意力（通道级多头注意力）
        - Eq.(7) A←B的交叉注意力（通道级多头注意力）
        - Eq.(8) 相关性门控融合（使用concat([A_sa, A_ca, rho])）
        - Eq.(9) 第一次残差连接：f_bar = w*A_ca + A_sa + A
        - Eq.(10) FFN + 残差：Out = H(f_bar) + A
    
    使用位置：
        - PaperHDGFFM.forward() 内部调用
        - 用于VIS和深度特征的融合，以及IR和VIS特征的融合
    
    输入：
        A: [B,C,H,W] 图像侧特征（如VIS特征 f_{g,l}）
        B: [B,C,H,W] 深度侧嵌入特征（如深度特征 \hat f_{d,l}）
    
    输出：
        Out: [B,C,H,W] 融合后的特征
        aux: 可选的辅助信息字典 {"w", "A_sa", "A_ca", "f_bar"}

    特点：
        - 通道级多头注意力（将通道作为每个像素的token）
        - 温度参数τ控制注意力分布
        - 两个残差连接（Eq.(9)和Eq.(10)）
    
    Args:
        channels: 通道数
        heads: 注意力头数
        tau: 温度参数（控制注意力分布的锐度）
        with_norm: 是否使用归一化（稳定训练）
    """
    def __init__(self, channels: int, heads: int = 4, tau: float = 1.0, with_norm: bool = True):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.c = channels
        self.h = heads
        self.d = channels // heads
        self.tau = tau
        self.with_norm = with_norm

        # Normalization before attention (stabilizes training)
        if with_norm:
            self.normA = nn.GroupNorm(8, channels)
            self.normB = nn.GroupNorm(8, channels)

        # Linear projections (1x1 conv) for SA(A) and CA(A<-B)
        # Eq.(5/6): SA on A
        self.qA_sa = nn.Conv2d(channels, channels, 1)
        self.kA_sa = nn.Conv2d(channels, channels, 1)
        self.vA_sa = nn.Conv2d(channels, channels, 1)
        # Eq.(7): CA A<-B
        self.qA = nn.Conv2d(channels, channels, 1)
        self.kB = nn.Conv2d(channels, channels, 1)
        self.vB = nn.Conv2d(channels, channels, 1)

        # Eq.(8): correlation-gated fusion, O(·)
        # concat([A_sa, A_ca, rho]) -> 1 channel weight, then σ
        self.gate = nn.Sequential(nn.Conv2d(channels * 2 + 1, 1, 1), nn.Sigmoid())

        # Eq.(10): H(·) = lightweight FFN
        self.ffn = DepthwiseFFN(channels)

    # ---- channel-wise MHA: [B,C,H,W] -> [B,C,H,W] ----
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B,H,W,h,d]
        B, C, H, W = x.shape
        return x.view(B, self.h, self.d, H, W).permute(0, 3, 4, 1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,H,W,h,d] -> [B,C,H,W]
        B, H, W, h, d = x.shape
        return x.permute(0, 3, 4, 1, 2).contiguous().view(B, h * d, H, W)

    def _mha_channel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink_active: Optional[torch.Tensor] = None,
        sink_mask: Optional[torch.Tensor] = None,
        var_p: float = 0.5,
    ) -> torch.Tensor:
        # per-pixel attention across channel tokens
        B, C, H, W = q.shape
        qh = self._split_heads(q)  # [B,H,W,h,d]
        kh = self._split_heads(k)
        vh = self._split_heads(v)

        # logits with temperature τ
        # [B,H,W,h,d,d]
        scale = (self.d ** -0.5) * self.tau
        logits = torch.einsum("byxhd,byxhf->byxhdf", qh, kh) * scale
        attn = logits.softmax(dim=-1)

        if sink_active is not None and sink_mask is not None:
            attn = self._apply_var_attention(attn, sink_active, sink_mask, var_p)

        out  = torch.einsum("byxhdf,byxhf->byxhd", attn, vh)  # [B,H,W,h,d]
        return self._merge_heads(out)  # [B,C,H,W]

    def _apply_var_attention(
        self,
        attn: torch.Tensor,
        sink_active: torch.Tensor,
        sink_mask: torch.Tensor,
        var_p: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if sink_mask.numel() == 0:
            return attn

        active = sink_active.to(dtype=attn.dtype)
        if not torch.any(active):
            return attn

        active = active.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,H,W,1,1,1]
        sink_mask_key = sink_mask.view(1, 1, 1, self.h, 1, self.d).to(attn.dtype)
        nonsink_mask_key = (~sink_mask).view(1, 1, 1, self.h, 1, self.d).to(attn.dtype)

        sink_cols = attn * sink_mask_key
        removed = sink_cols * var_p * active
        attn = attn - removed

        redistribute = removed.sum(dim=-1, keepdim=True)
        nonsink_cols = attn * nonsink_mask_key
        nonsink_sum = nonsink_cols.sum(dim=-1, keepdim=True)

        scale = torch.zeros_like(nonsink_sum)
        valid = nonsink_sum > eps
        scale[valid] = redistribute[valid] / (nonsink_sum[valid] + eps)
        attn = attn + nonsink_cols * scale
        return attn

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        return_aux: bool = False,
        sink_active: Optional[torch.Tensor] = None,
        var_sink_mask: Optional[torch.Tensor] = None,
        var_p: float = 0.5,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        if self.with_norm:
            A_, B_ = self.normA(A), self.normB(B)
        else:
            A_, B_ = A, B

        # Eq.(5/6): Self-Attention on A
        A_sa = self._mha_channel(self.qA_sa(A_), self.kA_sa(A_), self.vA_sa(A_))

        # Eq.(7): Cross-Attention A <- B
        A_ca = self._mha_channel(
            self.qA(A_), self.kB(B_), self.vB(B_),
            sink_active=sink_active,
            sink_mask=var_sink_mask,
            var_p=var_p,
        )

        # Eq.(8): correlation-gated fusion (rho from A_sa & A_ca)
        rho = _cosine_corr(A_sa, A_ca)                       # [B,1,H,W]
        w   = self.gate(torch.cat([A_sa, A_ca, rho], dim=1)) # [B,1,H,W]

        # Eq.(9): first residual add (note: +A here)
        f_bar = w * A_ca + A_sa + A

        # Eq.(10): FFN + residual to A (again add A)
        Out = self.ffn(f_bar) + A

        if return_aux:
            return Out, {"w": w, "A_sa": A_sa, "A_ca": A_ca, "f_bar": f_bar}
        return Out, None


# ------------------ Per-scale HDGFFM with depth-aware embedding E_l ------------------
class PaperHDGFFM(nn.Module):
    """
    单尺度HDGFFM模块：E_l (1x1卷积) + PaperCorrCrossAttn(A, B_emb)
    
    功能：
        - 深度感知嵌入：使用1x1卷积将深度特征B嵌入到与图像特征A相同的通道数
        - 相关性交叉注意力融合：使用PaperCorrCrossAttn进行特征融合
        - 用于VIS和深度特征的融合，以及IR和VIS特征的融合
    
    使用位置：
        - net.py (DualBranchUNet中使用)
        - vis_depth_fusions: VIS特征与深度特征的融合（vis2~vis5）
        - bottleneck_fusion: VIS特征与IR特征的融合（vis5和ir5）
    
    架构：
        - E_l: 1x1卷积，将深度特征通道数对齐到图像特征
        - PaperCorrCrossAttn: 基于相关性交叉注意力的融合

    Args:
        c_img: 图像侧特征A的通道数
        c_dep: 深度侧原始特征B的通道数（嵌入前）
        heads: 注意力头数（传递给PaperCorrCrossAttn）
        tau: 温度参数（传递给PaperCorrCrossAttn）
        with_norm: 是否使用归一化（传递给PaperCorrCrossAttn）
    """
    def __init__(
        self,
        c_img: int,
        c_dep: int,
        heads: int = 4,
        tau: float = 1.0,
        with_norm: bool = True,
        enable_var: bool = False,
    ):
        super().__init__()
        # E_l: 1x1 conv, no compression but aligns channels to A
        self.embed = nn.Conv2d(c_dep, c_img, 1)
        self.fuse  = PaperCorrCrossAttn(channels=c_img, heads=heads, tau=tau, with_norm=with_norm)

        # VAR settings
        self.enable_var = enable_var and utils.VAR_ENABLED and len(utils.VAR_SINK_DIMS) > 0
        sink_dims = torch.tensor(utils.VAR_SINK_DIMS, dtype=torch.long)
        if sink_dims.numel() > 0:
            sink_dims = torch.remainder(sink_dims, c_img).unique()
        self.register_buffer("var_sink_dims", sink_dims, persistent=False)

        head_dim = c_img // heads
        sink_mask_heads = torch.zeros(heads, head_dim, dtype=torch.bool)
        if sink_dims.numel() > 0:
            for idx in sink_dims.tolist():
                head_idx = min(heads - 1, idx // head_dim)
                pos = idx % head_dim
                sink_mask_heads[head_idx, pos] = True
        self.register_buffer("var_sink_mask_heads", sink_mask_heads, persistent=False)

        self.var_tau = utils.VAR_TAU
        self.var_p = utils.VAR_P

    def forward(self, A: torch.Tensor, B: torch.Tensor, return_aux: bool = False):
        B_emb = self.embed(B)
        sink_active = None
        sink_mask = None
        if self.enable_var and self.var_sink_dims.numel() > 0:
            sink_active = self._compute_sink_activity(B_emb)
            sink_mask = self.var_sink_mask_heads
        return self.fuse(
            A,
            B_emb,
            return_aux=return_aux,
            sink_active=sink_active,
            var_sink_mask=sink_mask,
            var_p=self.var_p,
        )

    def _compute_sink_activity(self, B_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = B_emb.shape
        x = B_emb.permute(0, 2, 3, 1).reshape(-1, C)
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-6)
        sink_vals = x[:, self.var_sink_dims]
        phi = sink_vals.max(dim=1).values / (rms.squeeze(1) + 1e-6)
        sink_tokens = (phi >= self.var_tau).view(B, H, W)
        return sink_tokens


# ------------------ Multi-scale pyramid wrapper ------------------
class PaperHDGFFMPyramid(nn.Module):
    """
    多尺度金字塔融合模块：在L个尺度上应用HDGFFM
    
    功能：
        - 在多个尺度上应用HDGFFM融合
        - 每个尺度有独立的E_l (1x1卷积)和CorrCrossAttn模块
        - 处理不同空间尺寸的特征对齐
    
    使用位置：
        - 目前未在net.py中使用（单尺度融合已足够）
        - 可用于需要多尺度融合的场景
    
    输入：
        A_list: [A1, A2, ..., AL] 图像侧特征列表（每个尺度的特征）
        B_list: [B1, B2, ..., BL] 深度侧特征列表（每个尺度的特征）
        （每个尺度内的空间尺寸需要一致）
    
    输出：
        outs: 融合后的特征列表
        auxs: 可选的辅助信息列表
    
    Args:
        c_imgs: 图像侧各尺度通道数列表
        c_deps: 深度侧各尺度通道数列表
        heads: 注意力头数
        tau: 温度参数
        with_norm: 是否使用归一化
    """
    def __init__(self, c_imgs: List[int], c_deps: List[int], heads: int = 4, tau: float = 1.0, with_norm: bool = True):
        super().__init__()
        assert len(c_imgs) == len(c_deps)
        self.L = len(c_imgs)
        self.blocks = nn.ModuleList([
            PaperHDGFFM(c_img=c_imgs[i], c_dep=c_deps[i], heads=heads, tau=tau, with_norm=with_norm)
            for i in range(self.L)
        ])

    def forward(
        self,
        A_list: List[torch.Tensor],
        B_list: List[torch.Tensor],
        return_aux: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[List[Dict[str, torch.Tensor]]]]:
        assert len(A_list) == len(B_list) == self.L
        outs: List[torch.Tensor] = []
        auxs: List[Dict[str, torch.Tensor]] = []
        for i, (blk, A, B) in enumerate(zip(self.blocks, A_list, B_list)):
            # If spatial sizes differ slightly, align B to A
            if A.shape[-2:] != B.shape[-2:]:
                B = F.interpolate(B, size=A.shape[-2:], mode="bilinear", align_corners=False)
            out_i, aux_i = blk(A, B, return_aux=return_aux)
            outs.append(out_i)
            if return_aux and aux_i is not None:
                auxs.append(aux_i)
        return (outs, auxs) if return_aux else (outs, None)


# ------------------ Quick self-check ------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Single-scale check
    B, C, H, W = 2, 64, 48, 64
    A = torch.randn(B, C, H, W, device=device)
    Bf= torch.randn(B, C, H, W, device=device)

    cca = PaperCorrCrossAttn(channels=C, heads=4, tau=1.0, with_norm=True).to(device)
    out, aux = cca(A, Bf, return_aux=True)
    print("[single] out:", out.shape, "| w:", aux["w"].shape)

    # Multi-scale check (channels can differ, E_l will align B to A)
    A_list = [torch.randn(B, 32, H, W, device=device),
              torch.randn(B, 64, H//2, W//2, device=device),
              torch.randn(B,128, H//4, W//4, device=device)]
    B_list = [torch.randn(B, 40, H, W, device=device),   # dep channels need not equal img channels
              torch.randn(B, 64, H//2, W//2, device=device),
              torch.randn(B,160, H//4, W//4, device=device)]

    pyramid = PaperHDGFFMPyramid(c_imgs=[32,64,128], c_deps=[40,64,160], heads=4, tau=1.0, with_norm=True).to(device)
    fused_list, _ = pyramid(A_list, B_list, return_aux=False)
    print("[pyramid] fused shapes:", [t.shape for t in fused_list])

