"""
DeepDetect - Shared Unified Preprocessing Module (Phase 2)
----------------------------------------------------------
Single source of truth for preparing image tensors for the CNN model.
Used consistently in:
  - Training (cnn_training.py)
  - Validation
  - Held-out testing
  - Streamlit dashboard inference (app.py)
  - Adversarial robustness evaluation (cnn_robustness.py)
"""

import cv2
import numpy as np

# Target input dimensions for Steganalysis CNN
IMG_SIZE = (224, 224)

# SRM / KV 5x5 High-Pass Filter Kernel (Fridrich & Kodovsky 2012)
KV_KERNEL = np.array([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8, -12, 8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1]
], dtype=np.float32) / 12.0

def compute_residual_channels(gray_f32):
    """
    Computes 3 spatial residual channels from a grayscale float32 image:
      Channel 0: Gaussian Residual: Gray - GaussianBlur(Gray) scaled by 1/4.0
      Channel 1: KV / SRM High-Pass Residual: filter2D(Gray, KV_KERNEL) scaled by 1/4.0
      Channel 2: Laplacian Residual: Laplacian(Gray) scaled by 1/8.0
    Returns:
      (H, W, 3) float32 array clipped to [-1.0, 1.0]
    """
    # Channel 0: Gaussian residual (sigma=1.0, 5x5 window)
    blurred = cv2.GaussianBlur(gray_f32, (5, 5), sigmaX=1.0)
    gauss_r = (gray_f32 - blurred) / 4.0

    # Channel 1: KV high-pass filter residual
    kv_r = cv2.filter2D(gray_f32, -1, KV_KERNEL) / 4.0

    # Channel 2: Laplacian 2nd order differential residual
    lap = cv2.Laplacian(gray_f32, cv2.CV_32F, ksize=3)
    lap_r = lap / 8.0

    # Stack into 3-channel feature map and clip to [-1.0, 1.0]
    residual_map = np.stack([gauss_r, kv_r, lap_r], axis=-1)
    residual_map = np.clip(residual_map, -1.0, 1.0).astype(np.float32)
    return residual_map

def preprocess_image_for_cnn(image, target_size=IMG_SIZE):
    """
    Unified CNN Preprocessing Function.
    
    Accepts:
      - image: numpy ndarray (BGR or RGB or Grayscale) or file path str
      - target_size: tuple (width, height), default (224, 224)
      
    Returns:
      - (H, W, 3) float32 tensor in range [-1.0, 1.0]
    """
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
        if img_bgr is None:
            raise ValueError(f"Unable to read image from path: {image}")
    else:
        img_bgr = image.copy()

    # If grayscale, convert to BGR first for consistent pipeline
    if len(img_bgr.shape) == 2:
        gray_f32 = img_bgr.astype(np.float32)
    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        gray_f32 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        # RGBA / BGRA
        gray_f32 = cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {img_bgr.shape}")

    # Resize grayscale image to target size using INTER_LINEAR
    if (gray_f32.shape[1], gray_f32.shape[0]) != target_size:
        gray_f32 = cv2.resize(gray_f32, target_size, interpolation=cv2.INTER_LINEAR)

    # Compute 3-channel residual map
    residual_tensor = compute_residual_channels(gray_f32)
    return residual_tensor

def get_residual_display_maps(image, target_size=IMG_SIZE):
    """
    Helper for Streamlit/Visualizations: returns individual displayable
    channels mapped to [0, 255] uint8.
    """
    tensor = preprocess_image_for_cnn(image, target_size=target_size)
    # Tensor is in [-1.0, 1.0], map to [0, 255]
    def to_uint8(ch):
        norm = (ch + 1.0) / 2.0
        return np.clip(norm * 255.0, 0, 255).astype(np.uint8)

    return {
        "gaussian_residual": to_uint8(tensor[:, :, 0]),
        "kv_residual": to_uint8(tensor[:, :, 1]),
        "laplacian_residual": to_uint8(tensor[:, :, 2]),
        "tensor": tensor
    }
