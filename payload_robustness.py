import os
import sys

os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(".", ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLEAN_SAMPLE = os.path.join("data", "clean", "clean_001.png")
if not os.path.exists(CLEAN_SAMPLE):
    # Fallback to any clean image or clean.jpg
    clean_folder = os.path.join("data", "clean")
    files = [f for f in os.listdir(clean_folder) if f.lower().endswith(('.png', '.jpg'))]
    CLEAN_SAMPLE = os.path.join(clean_folder, files[0])

def calculate_mse(img1, img2):
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))

def calculate_psnr(mse):
    if mse <= 1e-10:
        return 100.0  # Perfect reconstruction / identical
    return float(10 * np.log10((255.0 ** 2) / mse))

def calculate_ber(original_bits, extracted_bits):
    min_len = min(len(original_bits), len(extracted_bits))
    errors = int(np.count_nonzero(original_bits[:min_len] != extracted_bits[:min_len]))
    # Add difference in length if any as errors
    errors += abs(len(original_bits) - len(extracted_bits))
    ber = float(errors / len(original_bits)) if len(original_bits) > 0 else 0.0
    return errors, ber

def embed_lsb(image, message_bits):
    """Embeds message_bits into blue channel (channel 0) LSB without altering rest."""
    stego = image.copy()
    h, w, _ = stego.shape
    blue = stego[:, :, 0].flatten()
    n_bits = len(message_bits)
    blue[:n_bits] = (blue[:n_bits] & 254) | message_bits
    stego[:, :, 0] = blue.reshape(h, w)
    return stego

def extract_lsb(image, n_bits):
    """Extracts n_bits from blue channel LSB."""
    blue = image[:, :, 0].flatten()
    return blue[:n_bits] & 1

def run_payload_robustness():
    original_img = cv2.imread(CLEAN_SAMPLE)
    if original_img is None:
        raise FileNotFoundError(f"Cover image {CLEAN_SAMPLE} not found.")
        
    h, w, c = original_img.shape
    total_pixels = h * w
    # 1 bit per pixel blue channel capacity
    max_capacity_bits = total_pixels
    
    payload_rates = [0.001, 0.005, 0.01, 0.05, 0.10] # 0.1%, 0.5%, 1%, 5%, 10%
    
    results = []
    
    # Transformations to test on stego image
    def get_transformations(stego):
        t_dict = {}
        # 1. Untouched
        t_dict["Untouched"] = stego.copy()
        
        # 2. JPEG recompression (Q=90)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encimg = cv2.imencode('.jpg', stego, encode_param)
        t_dict["JPEG_Q90"] = cv2.imdecode(encimg, 1)
        
        # 3. Gaussian blur
        t_dict["Gaussian_Blur_3x3"] = cv2.GaussianBlur(stego, (3, 3), 0)
        
        # 4. Resizing (scale down 50% then restore)
        small = cv2.resize(stego, (max(1, w // 2), max(1, h // 2)))
        t_dict["Resize_and_Restore"] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 5. Gaussian noise (sigma=2)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 2, stego.shape)
        t_dict["Gaussian_Noise"] = np.clip(stego.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return t_dict

    # Deterministic pseudo-random bit generator for reproducible payload
    rng_payload = np.random.default_rng(12345)
    
    for rate in payload_rates:
        n_bits = int(max_capacity_bits * rate)
        n_bytes = n_bits // 8
        bpp = n_bits / total_pixels
        capacity_utilization = rate * 100.0
        
        # Generate random bit stream
        bits = rng_payload.integers(0, 2, size=n_bits, dtype=np.uint8)
        
        # Embed
        stego_img = embed_lsb(original_img, bits)
        
        # Calculate Stego image MSE and PSNR against original
        mse_val = calculate_mse(original_img, stego_img)
        psnr_val = calculate_psnr(mse_val)
        
        # Test across transformations
        transforms = get_transformations(stego_img)
        
        for t_name, trans_img in transforms.items():
            extracted_bits = extract_lsb(trans_img, n_bits)
            err_count, ber = calculate_ber(bits, extracted_bits)
            is_recovered = (err_count == 0)
            
            # Quality of transformed image relative to original
            t_mse = calculate_mse(original_img, trans_img)
            t_psnr = calculate_psnr(t_mse)
            
            results.append({
                "Payload_Rate_Target": f"{rate*100:.1f}%",
                "Payload_Bits": n_bits,
                "Payload_Bytes": n_bytes,
                "Bits_Per_Pixel": bpp,
                "Capacity_Utilization_Pct": capacity_utilization,
                "Stego_MSE": mse_val,
                "Stego_PSNR_dB": psnr_val,
                "Condition": t_name,
                "Transformed_MSE": t_mse,
                "Transformed_PSNR_dB": t_psnr,
                "Incorrect_Bits": err_count,
                "BER": ber,
                "Message_Recovered_Exactly": is_recovered
            })
            
    df_results = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, "payload_robustness.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Payload robustness testing completed. Saved {len(df_results)} rows to {out_csv}")
    
    # Plot 1: PSNR vs Payload Rate
    untouched_df = df_results[df_results["Condition"] == "Untouched"]
    plt.figure(figsize=(7, 4))
    plt.plot(untouched_df["Payload_Rate_Target"], untouched_df["Stego_PSNR_dB"], marker='o', color='b', linewidth=2)
    plt.title("Stego PSNR vs Payload Rate")
    plt.xlabel("Payload Capacity (%)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "payload_vs_psnr.png"), dpi=200)
    plt.close()
    
    # Plot 2: BER across Transformations
    plt.figure(figsize=(10, 5))
    for t_name in transforms.keys():
        sub = df_results[df_results["Condition"] == t_name]
        plt.plot(sub["Payload_Rate_Target"], sub["BER"], marker='s', label=t_name)
    plt.title("Bit Error Rate (BER) across Transformations & Payloads")
    plt.xlabel("Payload Capacity (%)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.ylim(-0.05, 0.6)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ber_vs_transformations.png"), dpi=200)
    plt.close()
    print("Saved plots to results/payload_vs_psnr.png and results/ber_vs_transformations.png")

if __name__ == "__main__":
    run_payload_robustness()
