# coding=utf-8
import os
import cv2
import numpy as np
import torch
from PIL import Image
import time
import glob
import re
from collections import OrderedDict
#from net import Net
# For ablation checkpoints, use exactly one matching import instead:
#from net_enhanceddown_cnn_ablation import Net
#from net_enhanceddown_fnet_ablation import Net
#from net_paperhdgffm_concat_ablation import Net

from net_no_depth_ablation import Net
#from net_no_crossattn_ablation import Net
#from net_depth_ir_only_ablation import Net
#from net_no_selfattn_ablation import Net
#from net_depth_vis_only_ablation import Net
import utils


def RGB2YCbCr(img):
    img = img * 255.0
    r, g, b = torch.split(img, 1, dim=1)
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    y = y / 255.0
    cb = -0.148 * r - 0.291 * g + 0.439 * b + 128
    cb = cb / 255.0
    cr = 0.439 * r - 0.368 * g - 0.071 * b + 128
    cr = cr / 255.0
    img = torch.cat([y, cb, cr], dim=1)
    return img


def YCbCr2RGB(img, img_Y):
    img = RGB2YCbCr(img) * 255
    y, cb, cr = torch.split(img, 1, dim=1)
    r = 1.164 * (img_Y * 255 - 16) + 1.596 * (cr - 128)
    r = r / 255.0
    g = 1.164 * (img_Y * 255 - 16) - 0.392 * (cb - 128) - 0.813 * (cr - 128)
    g = g / 255.0
    b = 1.164 * (img_Y * 255 - 16) + 2.017 * (cb - 128)
    b = b / 255.0
    img = torch.cat([b, g, r], dim=1)
    return img * 255


def prepare_data(dataset_dir, prefix=None):
    """Load and sort image files from directory."""
    patterns = [f"{prefix}*.jpg", f"{prefix}*.png", f"{prefix}*.bmp"] if prefix else ["*.jpg", "*.png", "*.bmp"]
    data = []
    for pat in patterns:
        data.extend(glob.glob(os.path.join(dataset_dir, pat)))

    if len(data) == 0:
        patt_desc = f"{prefix}*.jpg/.png/.bmp" if prefix else "*.jpg/.png/.bmp"
        raise FileNotFoundError(f"No images found in '{dataset_dir}' matching {patt_desc}.")

    def _extract_num(path):
        name = os.path.basename(path)
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 0

    data.sort(key=_extract_num)
    return data


def change(out):
    """Convert tensor to numpy array for saving."""
    out1 = out.cpu()
    out_img = out1.data[0]
    out_img = out_img.numpy()
    
    # Handle different tensor shapes
    if len(out_img.shape) == 2:
        # Already 2D (H, W)
        return out_img
    elif len(out_img.shape) == 3:
        # 3D tensor (C, H, W) -> (H, W, C)
        out_img = out_img.transpose(1, 2, 0)
        return out_img
    else:
        raise ValueError(f"Unexpected tensor shape: {out_img.shape}")


def count_parameters_in_MB(model):
    """Count model parameters in millions."""
    total = sum(p.numel() for name, p in model.named_parameters() if "auxiliary" not in name)
    print(f"Model parameters: {total / 1e6:.2f}M")


def load_image(x):
    """Load grayscale image (ensures single channel)."""
    imgA = Image.open(x)
    # Convert to grayscale if not already
    if imgA.mode != 'L':
        imgA = imgA.convert('L')
    imgA = np.asarray(imgA)
    imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float64)
    imgA = torch.from_numpy(imgA).float()
    imgA = imgA.unsqueeze(0)
    return imgA


def load_rgb(x):
    """Load RGB or grayscale image."""
    imgA = Image.open(x)
    imgA = np.asarray(imgA)
    
    # Handle grayscale images
    if len(imgA.shape) == 2:
        # Grayscale image: add channel dimension
        imgA = np.expand_dims(imgA, axis=2)
    elif len(imgA.shape) == 3 and imgA.shape[2] == 4:
        # RGBA image: convert to RGB
        imgA = imgA[:, :, :3]
    
    imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float64)
    imgA = torch.from_numpy(imgA).float()
    imgA = imgA.unsqueeze(0)
    return imgA


def RGB2Y(img):
    """Convert RGB to Y channel, or return grayscale as-is."""
    # Check number of channels
    num_channels = img.shape[1]
    
    if num_channels == 1:
        # Already grayscale, return as-is
        return img
    elif num_channels == 3:
        # RGB image, convert to Y channel
        r, g, b = torch.split(img, 1, dim=1)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y
    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}. Expected 1 (grayscale) or 3 (RGB).")


# Main execution
device = utils.get_device()
utils.ensure_dirs()
print(f"Using device: {device}")

# Load exactly the checkpoint configured in utils.MODEL_PATH.
model_paths = [
    utils.MODEL_PATH,
]

model = None
loaded = None
loaded_path = None

for model_path in model_paths:
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}, trying next...")
        continue
    
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        print(f"Warning: Model file is empty: {model_path}, trying next...")
        continue
    
    print(f"Attempting to load model from: {model_path} (size: {file_size} bytes)")
    try:
        loaded = torch.load(model_path, map_location=device)
        loaded_path = model_path
        print(f"Successfully loaded model from: {model_path}")
        break
    except EOFError:
        print(f"Error: Model file appears corrupted (EOFError): {model_path}")
        print(f"File size: {file_size} bytes. Trying next model file...")
        continue
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        continue

if loaded is None:
    raise FileNotFoundError(
        f"Failed to load model from any of the following paths:\n" +
        "\n".join(f"  - {p}" for p in model_paths) +
        "\nPlease ensure at least one valid model file exists."
    )

# Load model. Supports both legacy model-only checkpoints and full training checkpoints.
if isinstance(loaded, dict) and "model_state_dict" in loaded:
    model = Net().to(device)
    model.load_state_dict(loaded["model_state_dict"])
elif isinstance(loaded, OrderedDict):
    model = Net().to(device)
    model.load_state_dict(loaded)
else:
    model = loaded.to(device)

print(f"Model loaded successfully from: {loaded_path}")
count_parameters_in_MB(model)

# Prepare test data
image_IR_list = prepare_data(utils.TEST_IR_DIR)
image_VIS_list = prepare_data(utils.TEST_VI_DIR)
save_image_path = utils.OUTPUT_DIR
os.makedirs(save_image_path, exist_ok=True)

# Process images
all_time = []
min_len = min(len(image_IR_list), len(image_VIS_list))
if len(image_IR_list) != len(image_VIS_list):
    print(f"Warning: IR ({len(image_IR_list)}) and VIS ({len(image_VIS_list)}) counts differ. Processing first {min_len} pairs.")

for i in range(min_len):
    # Load images
    IR = load_image(image_IR_list[i])
    VIS = load_rgb(image_VIS_list[i])
    
    # Normalize to [0, 1]
    IR = (IR).to(device) / 255
    VIS = (VIS).to(device) / 255
    
    # Convert VIS to Y channel (handles both grayscale and RGB)
    VIS_y = RGB2Y(VIS)
    
    # Check if VIS is RGB or grayscale
    is_rgb = VIS.shape[1] == 3
    
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        Fused, Y, d_hat = model(IR, VIS_y)  # Now returns 3 values: fused, Y, d_hat
        
        # Handle output based on input type
        if is_rgb:
            # RGB input: convert back to RGB using YCbCr
            Fused_RGB = YCbCr2RGB(VIS, Fused)
            output_tensor = Fused_RGB.clamp(min=0, max=255)
        else:
            # Grayscale input: use fused result directly
            output_tensor = (Fused * 255).clamp(min=0, max=255)
        
        all_time.append(time.time() - start_time)

    # Convert to numpy and save
    out = change(output_tensor)
    
    # Ensure output is in correct format for saving
    if len(out.shape) == 2:
        # Grayscale: already correct, ensure uint8
        out = out.astype(np.uint8)
    elif len(out.shape) == 3 and out.shape[2] == 1:
        # Single channel: squeeze and convert to uint8
        out = out.squeeze(2).astype(np.uint8)
    elif len(out.shape) == 3 and out.shape[2] == 3:
        # RGB/BGR: YCbCr2RGB returns BGR format, so it's already correct for OpenCV
        # Ensure uint8
        out = out.astype(np.uint8)
    
    cv2.imwrite(os.path.join(save_image_path, str(i + 1) + '.bmp'), out)
    print(f'Fused image {i + 1} saved ({"RGB" if is_rgb else "Grayscale"} mode)')

print(f'Mean inference time: {np.mean(all_time):.4f}s, std: {np.std(all_time):.4f}s')
