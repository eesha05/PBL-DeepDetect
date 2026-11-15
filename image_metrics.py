"""
DeepDetect - Phase 6: Image Quality and Payload Metrics Module
--------------------------------------------------------------
Strict mathematical definitions for steganographic analysis:
  1. Imperceptibility:
     - MSE (Mean Squared Error, float64 casting to prevent underflow)
     - PSNR (Peak Signal-to-Noise Ratio in dB)
     - SSIM (Structural Similarity Index)
  2. Payload Capacity:
     - Payload Size (bits and bytes)
     - Payload Rate / Embedding Rate (bits per pixel, bpp)
     - Capacity Utilization (%)
     - Maximum Theoretical Capacity
  3. Message Recovery Robustness:
     - Bit Error Rate (BER = bit errors / total embedded bits)
     - Extraction fidelity under attacks
     - Clearly distinguished from detection accuracy
"""

import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim_fn

def calculate_mse(img1, img2):
    """Computes Mean Squared Error with float64 casting."""
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def calculate_psnr(img1, img2):
    """Computes Peak Signal-to-Noise Ratio (dB)."""
    mse = calculate_mse(img1, img2)
    if mse <= 1e-12:
        return 100.0  # Identical images
    return float(10 * np.log10((255.0 ** 2) / mse))

def calculate_ssim(img1, img2):
    """Computes Structural Similarity Index (SSIM)."""
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        return float(ssim_fn(img1, img2, channel_axis=2, data_range=255))
    else:
        return float(ssim_fn(img1, img2, data_range=255))

def calculate_ber(original_bits, extracted_bits):
    """
    Computes Bit Error Rate between original embedded bits and recovered bits.
    Handles length discrepancies safely.
    """
    orig = np.array(original_bits, dtype=np.uint8)
    extr = np.array(extracted_bits, dtype=np.uint8)
    
    n_orig = len(orig)
    if n_orig == 0:
        return 0, 0.0
        
    common_len = min(n_orig, len(extr))
    errors = int(np.count_nonzero(orig[:common_len] != extr[:common_len]))
    # Add length discrepancy as bit errors
    errors += abs(n_orig - len(extr))
    ber = float(errors / n_orig)
    return errors, ber

def calculate_payload_metrics(image, embedded_bits_count):
    """
    Calculates payload and embedding capacity metrics for 1-bpp LSB embedding:
      - Payload Size in bits and bytes
      - Payload Rate in bits per pixel (bpp)
      - Maximum Theoretical Capacity
      - Capacity Utilization (%)
    """
    h, w = image.shape[:2]
    total_pixels = h * w
    max_capacity_bits = total_pixels  # 1 bit per pixel (Blue channel)
    
    payload_rate_bpp = float(embedded_bits_count / total_pixels) if total_pixels > 0 else 0.0
    utilization_pct = float((embedded_bits_count / max_capacity_bits) * 100.0) if max_capacity_bits > 0 else 0.0
    
    return {
        "image_width": w,
        "image_height": h,
        "total_pixels": total_pixels,
        "max_capacity_bits": max_capacity_bits,
        "max_capacity_bytes": max_capacity_bits // 8,
        "embedded_bits": embedded_bits_count,
        "embedded_bytes": embedded_bits_count // 8,
        "payload_rate_bpp": payload_rate_bpp,
        "capacity_utilization_pct": utilization_pct
    }

def compute_all_image_metrics(cover_img, stego_img, original_bits=None, extracted_bits=None):
    """
    Comprehensive unified metric extraction returning all 8 essential parameters.
    """
    mse = calculate_mse(cover_img, stego_img)
    psnr = calculate_psnr(cover_img, stego_img)
    ssim = calculate_ssim(cover_img, stego_img)
    
    # Payload
    n_bits = len(original_bits) if original_bits is not None else 0
    payload = calculate_payload_metrics(cover_img, n_bits)
    
    # BER
    if original_bits is not None and extracted_bits is not None:
        errors, ber = calculate_ber(original_bits, extracted_bits)
    else:
        errors, ber = 0, 0.0
        
    return {
        "MSE": mse,
        "PSNR_dB": psnr,
        "SSIM": ssim,
        "Payload_Bits": n_bits,
        "Payload_Bytes": n_bits // 8,
        "Payload_Rate_bpp": payload["payload_rate_bpp"],
        "Capacity_Utilization_Pct": payload["capacity_utilization_pct"],
        "Max_Capacity_Bits": payload["max_capacity_bits"],
        "Bit_Errors": errors,
        "BER": ber
    }

if __name__ == "__main__":
    # Self-validation test
    clean_sample_path = os.path.join("data", "clean", "clean_001.png")
    stego_sample_path = os.path.join("data", "stego", "stego_001.png")
    if os.path.exists(clean_sample_path) and os.path.exists(stego_sample_path):
        c_img = cv2.imread(clean_sample_path)
        s_img = cv2.imread(stego_sample_path)
        metrics = compute_all_image_metrics(c_img, s_img)
        print("Self-Test Metric Calculation for clean_001.png <-> stego_001.png:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
