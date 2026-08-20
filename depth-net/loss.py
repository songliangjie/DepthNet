import torch
import torch.nn as nn
import torch.nn.functional as Fnn


# =====================
# Image Fusion Loss Functions
# =====================

def _get_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Generate 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _gaussian_blur(input_tensor: torch.Tensor, kernel_size: int = 5, sigma: float = 3.0) -> torch.Tensor:
    """Apply Gaussian blur to input tensor."""
    channels = input_tensor.shape[1]
    kernel_1d = _get_gaussian_kernel(kernel_size, sigma).to(input_tensor.device, input_tensor.dtype)
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,K,K]
    kernel = kernel_2d.repeat(channels, 1, 1, 1)  # [C,1,K,K]
    return Fnn.conv2d(input_tensor, kernel, padding=kernel_size // 2, groups=channels)


def _sobel_xy(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude (horizontal + vertical)."""
    device, dtype = x.device, x.dtype
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    C = x.shape[1]
    kx = kx.repeat(C, 1, 1, 1)
    ky = ky.repeat(C, 1, 1, 1)
    gx = Fnn.conv2d(x, kx, padding=1, groups=C)
    gy = Fnn.conv2d(x, ky, padding=1, groups=C)
    return torch.abs(gx) + torch.abs(gy)


def _unsharp_mask(
    x: torch.Tensor,
    lambda_gain: float = 1.5,
    sigma: float = 1.0,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    Texture enhancement function E(·) based on Unsharp Masking (USM):

        E(x) = x + λ_gain * (x - G_σ(x))

    where G_σ(·) is a Gaussian blur with standard deviation σ and kernel size K.
    The term (x - G_σ(x)) extracts high-frequency details, which are amplified
    by λ_gain and added back to x.
    """
    g = _gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)
    return x + lambda_gain * (x - g)


def _low_freq(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Extract low-frequency component using Gaussian blur (approximates Laplacian Pyramid)."""
    return _gaussian_blur(x, kernel_size=5, sigma=sigma)


def L_GradE(image_A: torch.Tensor, image_B: torch.Tensor, image_fused: torch.Tensor,
            lamb: float = 1.5, sigma1: float = 1.0, sigma2: float = 0.0) -> torch.Tensor:
    """
    Gradient Enhancement Loss using Unsharp Masking (USM) and Sobel operators.
    
    E(x) = x + λ_gain * (x - G_σ(x))
    
    Args:
        image_A: Source image A (IR) [B, C, H, W]
        image_B: Source image B (VIS) [B, C, H, W]
        image_fused: Fused image [B, C, H, W]
        lamb: λ_gain in USM (default: 1.5)
        sigma1: Gaussian σ in USM (default: 1.0)
        sigma2: (unused, kept for backward compatibility)
    
    Returns:
        L1 loss between fused gradient and target gradient
    """
    # Extract Y channel (first channel)
    Y = lambda t: t[:, :1, :, :]
    A, B, Fused = map(Y, (image_A, image_B, image_fused))
    
    # Unsharp Masking enhancement (texture / edge enhancement)
    A_e = _unsharp_mask(A, lambda_gain=lamb, sigma=sigma1, kernel_size=3)
    B_e = _unsharp_mask(B, lambda_gain=lamb, sigma=sigma1, kernel_size=3)
    
    # Compute Sobel gradients
    gA = _sobel_xy(A_e)
    gB = _sobel_xy(B_e)
    gF = _sobel_xy(Fused)
    
    # Target: maximum gradient from both source images
    target = torch.max(gA, gB)
    
    return Fnn.l1_loss(gF, target)


# ===================================================================================
# ABLATION STUDY: Baseline L_GradE (without USM enhancement), kept for comparison.
# ===================================================================================
# def L_GradE(image_A: torch.Tensor, image_B: torch.Tensor, image_fused: torch.Tensor,
#             lamb: float = 1.5, sigma1: float = 1.0, sigma2: float = 0.0) -> torch.Tensor:
#     """
#     Baseline Gradient Loss without USM enhancement.
#
#     Directly computes Sobel gradients on original images, then L1 loss between
#     fused gradient and max(IR, VIS) gradient.
#     """
#     Y = lambda t: t[:, :1, :, :]
#     A, B, Fused = map(Y, (image_A, image_B, image_fused))
#
#     gA = _sobel_xy(A)
#     gB = _sobel_xy(B)
#     gF = _sobel_xy(Fused)
#
#     target = torch.max(gA, gB)
#     return Fnn.l1_loss(gF, target)


def L_Int(image_A: torch.Tensor, image_B: torch.Tensor, image_fused: torch.Tensor) -> torch.Tensor:
    """
    Intensity Loss: ensures fused image preserves maximum intensity from source images.
    
    Args:
        image_A: Source image A (IR) [B, C, H, W]
        image_B: Source image B (VIS) [B, C, H, W]
        image_fused: Fused image [B, C, H, W]
    
    Returns:
        L1 loss between fused intensity and target intensity
    """
    image_A_Y = image_A[:, :1, :, :]
    image_B_Y = image_B[:, :1, :, :]
    image_fused_Y = image_fused[:, :1, :, :]
    
    # Target: maximum intensity from both source images
    x_in_max = torch.max(image_A_Y, image_B_Y)
    
    return Fnn.l1_loss(x_in_max, image_fused_Y)


def L_Mask(image_A: torch.Tensor, image_B: torch.Tensor, image_fused: torch.Tensor, sigma_low: float = 1.0) -> torch.Tensor:
    """
    Mask Loss: low-frequency component fusion using adaptive mask.
    
    Args:
        image_A: Source image A (IR) [B, C, H, W]
        image_B: Source image B (VIS) [B, C, H, W]
        image_fused: Fused image [B, C, H, W]
        sigma_low: Gaussian sigma for low-frequency extraction (default: 1.0)
    
    Returns:
        MSE loss between fused low-frequency and target low-frequency
    """
    # Extract low-frequency components
    A_low = _low_freq(image_A, sigma=sigma_low)
    B_low = _low_freq(image_B, sigma=sigma_low)
    F_low = _low_freq(image_fused, sigma=sigma_low)
    
    # Adaptive target: max + ReLU(A - B) to preserve IR intensity when higher
    target_low = torch.max(A_low, B_low) + torch.relu(A_low - B_low)
    
    return Fnn.mse_loss(F_low, target_low)
