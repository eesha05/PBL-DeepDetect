"""
DeepDetect - Phase 5: Image Transformation Experiments
------------------------------------------------------
Applies controlled transformations on the strictly held-out test set
(from results/cnn_test_set_metadata.csv) and evaluates both:
  - Machine Learning Detectors: Edge LR, Residual LR, Combined LR
  - Deep Learning Detector: Residual CNN (via unified cnn_preprocessing)

Measures:
  - Classification accuracy & sensitivity/recall across transformations
  - Model prediction shifts and evasion rate (Stego -> Clean)
  - Image quality degradation (MSE, PSNR)
  - Explicit distinction between message recovery vs detector robustness vs evasion
"""

import os
import sys

os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(".", ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ["KERAS_BACKEND"] = "jax"

import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import keras
from keras import models

from residual_analysis import extract_all_features
from cnn_preprocessing import preprocess_image_for_cnn

CLEAN_DIR = os.path.join("data", "clean")
STEGO_DIR = os.path.join("data", "stego")
MODELS_DIR = "models"
RESULTS_DIR = "results"
TRANS_OUT_DIR = os.path.join(RESULTS_DIR, "transformed_images")
os.makedirs(TRANS_OUT_DIR, exist_ok=True)

def apply_transformation(image_bgr, trans_type):
    """
    Applies image transformation and returns transformed BGR image.
    1. none: identity
    2. jpeg: JPEG recompression (Q=75)
    3. blur: Gaussian blur (5x5, sigma=1.0)
    4. resize: Downsample 50% with bilinear interpolation and restore
    5. noise: Additive Gaussian noise (std=5)
    6. crop: 80% center crop and resize back to original dimension
    """
    h, w, c = image_bgr.shape
    if trans_type == "none":
        return image_bgr.copy()
    elif trans_type == "jpeg":
        _, enc = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return cv2.imdecode(enc, 1)
    elif trans_type == "blur":
        return cv2.GaussianBlur(image_bgr, (5, 5), 1.0)
    elif trans_type == "resize":
        small = cv2.resize(image_bgr, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif trans_type == "noise":
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 5, image_bgr.shape)
        return np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif trans_type == "crop":
        ch, cw = int(h * 0.8), int(w * 0.8)
        top = (h - ch) // 2
        left = (w - cw) // 2
        cropped = image_bgr[top:top+ch, left:left+cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        return image_bgr.copy()

def compute_quality_metrics(orig_img, trans_img):
    mse = float(np.mean((orig_img.astype(np.float64) - trans_img.astype(np.float64)) ** 2))
    psnr = float(10 * np.log10((255.0 ** 2) / mse)) if mse > 0 else 100.0
    return mse, psnr

def run_transformation_experiments():
    print("=" * 60)
    print("PHASE 5: IMAGE TRANSFORMATION AND EVASION EXPERIMENTS")
    print("=" * 60)

    # 1. Load ML models
    edge_model = joblib.load(os.path.join(MODELS_DIR, "Edge_Only_LogisticRegression.joblib"))
    res_model = joblib.load(os.path.join(MODELS_DIR, "Residual_Only_LogisticRegression.joblib"))
    comb_model = joblib.load(os.path.join(MODELS_DIR, "Combined_LogisticRegression.joblib"))

    # 2. Load CNN model if available
    cnn_path = os.path.join(MODELS_DIR, "cnn_model.keras")
    cnn_model = models.load_model(cnn_path) if os.path.exists(cnn_path) else None

    # Load feature column definitions
    feat_df = pd.read_csv(os.path.join(RESULTS_DIR, "deepdetect_all_features.csv"))
    edge_cols = ["canny_edge_density", "sobel_mean", "sobel_std", "laplacian_mean", "laplacian_variance"]
    res_cols = [c for c in feat_df.columns if c.startswith("residual_") or c.startswith("mean_abs") or c.startswith("res_hist_")]
    comb_cols = edge_cols + res_cols

    # 3. Load Held-Out Test Set from metadata to prevent evaluation mismatch
    meta_df = pd.read_csv(os.path.join(RESULTS_DIR, "cnn_test_set_metadata.csv"))
    print(f"Loaded held-out test set: {len(meta_df)} images ({sum(meta_df['true_label']==0)} clean, {sum(meta_df['true_label']==1)} stego)")

    test_items = []
    for _, row in meta_df.iterrows():
        fname = row["filename"]
        label = int(row["true_label"])
        folder = STEGO_DIR if label == 1 else CLEAN_DIR
        test_items.append((os.path.join(folder, fname), label, fname))

    transformations = ["none", "jpeg", "blur", "resize", "noise", "crop"]
    all_results = []
    detailed_evasion_records = []

    for t_type in transformations:
        y_true = []
        preds_edge, preds_res, preds_comb, preds_cnn = [], [], [], []
        mses, psnrs = [], []

        for img_path, label, fname in test_items:
            img = cv2.imread(img_path)
            if img is None:
                continue

            trans_img = apply_transformation(img, t_type)

            # Save sample transformed file
            sample_save_path = os.path.join(TRANS_OUT_DIR, f"{t_type}_{fname}")
            if not os.path.exists(sample_save_path):
                cv2.imwrite(sample_save_path, trans_img)

            # Quality metrics
            m_val, p_val = compute_quality_metrics(img, trans_img)
            mses.append(m_val)
            psnrs.append(p_val)

            # Feature extraction for Logistic Regression models
            gray = cv2.cvtColor(cv2.resize(trans_img, (224, 224)), cv2.COLOR_BGR2GRAY)
            feats = extract_all_features(gray)

            vec_edge = np.array([[feats[c] for c in edge_cols]])
            vec_res = np.array([[feats[c] for c in res_cols]])
            vec_comb = np.array([[feats[c] for c in comb_cols]])

            p_edge = int(edge_model.predict(vec_edge)[0])
            p_res = int(res_model.predict(vec_res)[0])
            p_comb = int(comb_model.predict(vec_comb)[0])

            # CNN prediction using unified preprocessing
            if cnn_model is not None:
                cnn_tensor = preprocess_image_for_cnn(trans_img)
                cnn_in = np.expand_dims(cnn_tensor, axis=0)
                cnn_prob = float(cnn_model.predict(cnn_in, verbose=0)[0][0])
                p_cnn = int(cnn_prob >= 0.5)
            else:
                p_cnn = 0

            y_true.append(label)
            preds_edge.append(p_edge)
            preds_res.append(p_res)
            preds_comb.append(p_comb)
            preds_cnn.append(p_cnn)

            detailed_evasion_records.append({
                "Transformation": t_type,
                "Filename": fname,
                "True_Label": label,
                "Pred_Edge": p_edge,
                "Pred_Residual": p_res,
                "Pred_Combined": p_comb,
                "Pred_CNN": p_cnn,
                "MSE": m_val,
                "PSNR_dB": p_val
            })

        y_true = np.array(y_true)
        avg_mse = float(np.mean(mses))
        avg_psnr = float(np.mean(psnrs))

        models_list = [
            ("Edge_Only_LR", preds_edge),
            ("Residual_Only_LR", preds_res),
            ("Combined_LR", preds_comb)
        ]
        if cnn_model is not None:
            models_list.append(("Residual_CNN", preds_cnn))

        for m_name, preds in models_list:
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
            evasion_rate = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

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
                "Stego_Evasion_Rate": evasion_rate,
                "Evasion_Count": fn,
                "Total_Stego_Tested": int(np.sum(y_true == 1)),
                "Total_Clean_Tested": int(np.sum(y_true == 0))
            })

    res_df = pd.DataFrame(all_results)
    out_csv = os.path.join(RESULTS_DIR, "transformation_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved transformation summary to {out_csv}")
    print(res_df[["Transformation", "Model", "Accuracy", "Sensitivity_Recall", "False_Positive_Rate", "Average_PSNR_dB"]])

    # Plot detector robustness comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(data=res_df, x="Transformation", y="Accuracy", hue="Model")
    plt.title("Steganalysis Model Robustness Across Image Transformations")
    plt.ylabel("Accuracy on Held-Out Test Set")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformation_accuracy_impact.png"), dpi=200)
    plt.close()
    print("Saved plot -> results/transformation_accuracy_impact.png")

if __name__ == "__main__":
    run_transformation_experiments()
