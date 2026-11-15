import os
import sys

os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(".", ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from residual_analysis import (
    load_and_preprocess, extract_edge_features, extract_residual_features,
    extract_all_features, CLEAN_DIR, STEGO_DIR, RESULTS_DIR, MODELS_DIR
)

TRANS_OUT_DIR = os.path.join(RESULTS_DIR, "transformed_images")
os.makedirs(TRANS_OUT_DIR, exist_ok=True)

def apply_transformation(image, trans_type):
    """
    Applies image transformation and returns transformed image in uint8.
    Supported types:
    1. jpeg: JPEG recompression (Quality = 75)
    2. blur: Gaussian blur (5x5, sigma=1)
    3. resize: Resize down 50% then restore
    4. noise: Add Gaussian noise (std=5)
    5. crop: Center crop 80% and resize back to original dimension
    """
    h, w, c = image.shape
    if trans_type == "jpeg":
        _, enc = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return cv2.imdecode(enc, 1)
    elif trans_type == "blur":
        return cv2.GaussianBlur(image, (5, 5), 1.0)
    elif trans_type == "resize":
        small = cv2.resize(image, (max(1, w // 2), max(1, h // 2)))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif trans_type == "noise":
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 5, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif trans_type == "crop":
        ch, cw = int(h * 0.8), int(w * 0.8)
        top = (h - ch) // 2
        left = (w - cw) // 2
        cropped = image[top:top+ch, left:left+cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        return image.copy()

def compute_quality_metrics(orig_img, trans_img):
    """Computes MSE and PSNR."""
    mse = float(np.mean((orig_img.astype(np.float64) - trans_img.astype(np.float64)) ** 2))
    psnr = float(10 * np.log10((255.0 ** 2) / mse)) if mse > 0 else 100.0
    return mse, psnr

def run_transformation_experiments():
    # Load trained models
    edge_model = joblib.load(os.path.join(MODELS_DIR, "Edge_Only_LogisticRegression.joblib"))
    res_model = joblib.load(os.path.join(MODELS_DIR, "Residual_Only_LogisticRegression.joblib"))
    comb_model = joblib.load(os.path.join(MODELS_DIR, "Combined_LogisticRegression.joblib"))
    
    # Feature columns order from deepdetect_all_features.csv
    features_csv = os.path.join(RESULTS_DIR, "deepdetect_all_features.csv")
    feat_df = pd.read_csv(features_csv)
    
    edge_cols = ["canny_edge_density", "sobel_mean", "sobel_std", "laplacian_mean", "laplacian_variance"]
    res_cols = [c for c in feat_df.columns if c.startswith("residual_") or c.startswith("mean_abs") or c.startswith("res_hist_")]
    comb_cols = edge_cols + res_cols
    
    # Choose 25 clean and 25 stego test samples
    clean_files = sorted([f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(('.png', '.jpg'))])[:25]
    stego_files = sorted([f for f in os.listdir(STEGO_DIR) if f.lower().endswith(('.png', '.jpg'))])[:25]
    
    test_items = [(os.path.join(CLEAN_DIR, f), 0, "clean", f) for f in clean_files] + \
                 [(os.path.join(STEGO_DIR, f), 1, "stego", f) for f in stego_files]
                 
    transformations = ["none", "jpeg", "blur", "resize", "noise", "crop"]
    
    all_results = []
    
    for t_type in transformations:
        y_true = []
        preds_edge = []
        preds_res = []
        preds_comb = []
        mses = []
        psnrs = []
        
        for img_path, label, img_type, fname in test_items:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            trans_img = apply_transformation(img, t_type) if t_type != "none" else img
            
            # Save transformed sample once per type
            sample_save_path = os.path.join(TRANS_OUT_DIR, f"{t_type}_{fname}")
            if not os.path.exists(sample_save_path):
                cv2.imwrite(sample_save_path, trans_img)
                
            # Quality metrics
            mse_val, psnr_val = compute_quality_metrics(img, trans_img)
            mses.append(mse_val)
            psnrs.append(psnr_val)
            
            # Extract features on transformed grayscale
            gray = cv2.cvtColor(cv2.resize(trans_img, (224, 224)), cv2.COLOR_BGR2GRAY)
            feats = extract_all_features(gray)
            
            vec_edge = np.array([[feats[c] for c in edge_cols]])
            vec_res = np.array([[feats[c] for c in res_cols]])
            vec_comb = np.array([[feats[c] for c in comb_cols]])
            
            p_edge = int(edge_model.predict(vec_edge)[0])
            p_res = int(res_model.predict(vec_res)[0])
            p_comb = int(comb_model.predict(vec_comb)[0])
            
            y_true.append(label)
            preds_edge.append(p_edge)
            preds_res.append(p_res)
            preds_comb.append(p_comb)
            
        y_true = np.array(y_true)
        avg_mse = float(np.mean(mses))
        avg_psnr = float(np.mean(psnrs))
        
        models_preds = [
            ("Edge_Only", preds_edge),
            ("Residual_Only", preds_res),
            ("Combined", preds_comb)
        ]
        
        for m_name, preds in models_preds:
            preds = np.array(preds)
            acc = float(accuracy_score(y_true, preds))
            prec = float(precision_score(y_true, preds, zero_division=0))
            rec = float(recall_score(y_true, preds, zero_division=0))
            f1 = float(f1_score(y_true, preds, zero_division=0))
            
            tn = int(np.sum((y_true == 0) & (preds == 0)))
            fp = int(np.sum((y_true == 0) & (preds == 1)))
            fn = int(np.sum((y_true == 1) & (preds == 0)))
            tp = int(np.sum((y_true == 1) & (preds == 1)))
            
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
            
            # Evasion count (Stego correctly identified as stego originally, but now predicted clean 0)
            evasion_count = fn
            
            all_results.append({
                "Transformation": t_type,
                "Model": m_name,
                "Average_MSE": avg_mse,
                "Average_PSNR_dB": avg_psnr,
                "Accuracy": acc,
                "Sensitivity_Recall": rec,
                "Specificity": spec,
                "Precision": prec,
                "F1_Score": f1,
                "False_Positive_Rate": fpr,
                "Evasion_Count": evasion_count,
                "Total_Stego_Tested": int(np.sum(y_true == 1)),
                "Total_Clean_Tested": int(np.sum(y_true == 0))
            })
            
    res_df = pd.DataFrame(all_results)
    out_csv = os.path.join(RESULTS_DIR, "transformation_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved transformation and evasion results to {out_csv}")
    print(res_df[["Transformation", "Model", "Accuracy", "Sensitivity_Recall", "False_Positive_Rate", "Average_PSNR_dB"]])
    
    # Visualization: Accuracy degradation under transformations
    plt.figure(figsize=(10, 5))
    sns.barplot(data=res_df, x="Transformation", y="Accuracy", hue="Model")
    plt.title("Steganalysis Detector Robustness Under Image Transformations")
    plt.ylabel("Classification Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformation_accuracy_impact.png"), dpi=200)
    plt.close()
    print("Saved plot to results/transformation_accuracy_impact.png")

if __name__ == "__main__":
    run_transformation_experiments()
