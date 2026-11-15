import os
import sys

# Configure matplotlib configdir to current project directory to prevent Windows sandbox permission error
os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(".", ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

CLEAN_DIR = os.path.join("data", "clean")
STEGO_DIR = os.path.join("data", "stego")
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_preprocess(image_path, image_size=(224, 224)):
    """Loads image, converts to RGB and Grayscale, preserves image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(image_rgb, image_size)
    grayscale = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
    return resized_rgb, grayscale

def compute_residual(grayscale_image, ksize=(5, 5), sigma=0):
    """
    Computes noise residual: R = I - GaussianBlur(I).
    Returns both signed float residual and normalized uint8 residual for visualization.
    """
    blurred = cv2.GaussianBlur(grayscale_image, ksize, sigma)
    # Signed residual to capture subtle variance
    residual_float = grayscale_image.astype(np.float64) - blurred.astype(np.float64)
    # Normalized uint8 residual
    residual_uint8 = cv2.absdiff(grayscale_image, blurred)
    return residual_float, residual_uint8

def extract_edge_features(grayscale_image):
    """Retains and standardizes existing edge features."""
    # Sobel
    sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Laplacian
    laplacian = np.absolute(cv2.Laplacian(grayscale_image, cv2.CV_64F))
    laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Canny
    canny = cv2.Canny(grayscale_image, threshold1=50, threshold2=150)
    canny_density = np.count_nonzero(canny) / (canny.shape[0] * canny.shape[1])
    
    return {
        "canny_edge_density": float(canny_density),
        "sobel_mean": float(np.mean(gradient_norm)),
        "sobel_std": float(np.std(gradient_norm)),
        "laplacian_mean": float(np.mean(laplacian_norm)),
        "laplacian_variance": float(np.var(laplacian_norm))
    }

def extract_residual_features(grayscale_image):
    """Extracts comprehensive statistical and high-frequency features from residual image."""
    res_float, res_uint8 = compute_residual(grayscale_image)
    
    res_mean = float(np.mean(res_float))
    res_std = float(np.std(res_float))
    res_var = float(np.var(res_float))
    mean_abs_res = float(np.mean(np.abs(res_float)))
    
    # Higher order moments
    res_skew = float(skew(res_float.flatten()))
    res_kurt = float(kurtosis(res_float.flatten()))
    
    # Residual histogram features (10 bins over abs residual)
    hist, _ = np.histogram(res_uint8, bins=10, range=(0, 255), density=True)
    
    features = {
        "residual_mean": res_mean,
        "residual_std": res_std,
        "residual_variance": res_var,
        "mean_absolute_residual": mean_abs_res,
        "residual_skewness": res_skew,
        "residual_kurtosis": res_kurt,
    }
    for i, h_val in enumerate(hist):
        features[f"res_hist_bin_{i}"] = float(h_val)
        
    return features

def extract_all_features(grayscale_image):
    """Combines edge and residual feature sets."""
    features = {}
    features.update(extract_edge_features(grayscale_image))
    features.update(extract_residual_features(grayscale_image))
    return features

def extract_dataset_features():
    """Extracts features across clean and stego sets preventing data leakage with image_id mapping."""
    records = []
    
    groups = [
        (CLEAN_DIR, 0, "clean"),
        (STEGO_DIR, 1, "stego")
    ]
    
    for folder, label, img_type in groups:
        if not os.path.exists(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            path = os.path.join(folder, fname)
            try:
                _, gray = load_and_preprocess(path)
                feats = extract_all_features(gray)
                feats["filename"] = fname
                feats["label"] = label
                feats["image_type"] = img_type
                
                # Extract image ID for pairing (e.g. clean_001.png or stego_001.png -> '001')
                base_name = os.path.splitext(fname)[0]
                digits = "".join([c for c in base_name if c.isdigit()])
                feats["group_id"] = digits if digits else base_name
                
                records.append(feats)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                
    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "deepdetect_all_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"Extracted {len(df)} records. Saved features to {csv_path}")
    return df

def train_and_evaluate_models():
    """Trains Edge-only, Residual-only, and Combined Logistic Regression models with zero data leakage."""
    df = extract_dataset_features()
    if df.empty:
        raise ValueError("No images processed!")
        
    edge_cols = ["canny_edge_density", "sobel_mean", "sobel_std", "laplacian_mean", "laplacian_variance"]
    residual_cols = [c for c in df.columns if c.startswith("residual_") or c.startswith("mean_abs") or c.startswith("res_hist_")]
    combined_cols = edge_cols + residual_cols
    
    # Group-based split based on group_id (original cover pairing)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(df, df["label"], groups=df["group_id"]))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    print(f"Train samples: {len(train_df)} ({len(train_df['group_id'].unique())} unique scenes)")
    print(f"Test samples: {len(test_df)} ({len(test_df['group_id'].unique())} unique scenes)")
    
    models_config = {
        "Edge_Only_LogisticRegression": edge_cols,
        "Residual_Only_LogisticRegression": residual_cols,
        "Combined_LogisticRegression": combined_cols
    }
    
    evaluation_summary = []
    
    plt.figure(figsize=(8, 6))
    
    for model_name, feature_cols in models_config.items():
        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Save model
        model_file = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        joblib.dump(pipeline, model_file)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0) # Sensitivity
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        try:
            auc = roc_auc_score(y_test, y_prob)
            fpr_vals, tpr_vals, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr_vals, tpr_vals, label=f"{model_name.replace('_', ' ')} (AUC = {auc:.3f})")
        except Exception:
            auc = 0.5
            
        evaluation_summary.append({
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Sensitivity_Recall": rec,
            "Specificity": specificity,
            "F1_Score": f1,
            "FPR": fpr,
            "ROC_AUC": auc,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Features_Count": len(feature_cols)
        })
        
        # Save individual confusion matrix plot
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Clean (0)", "Stego (1)"],
                    yticklabels=["Clean (0)", "Stego (1)"])
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png"), dpi=200)
        plt.close()

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve_comparison.png"), dpi=300)
    plt.close()
    
    summary_df = pd.DataFrame(evaluation_summary)
    comparison_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    summary_df.to_csv(comparison_csv, index=False)
    print("\n--- MODEL COMPARISON SUMMARY ---")
    print(summary_df[["Model", "Accuracy", "Precision", "Sensitivity_Recall", "Specificity", "F1_Score", "ROC_AUC"]])
    print(f"\nSaved comparison to {comparison_csv}")
    
    # Save visual comparison bar chart
    plt.figure(figsize=(10, 5))
    metrics_to_plot = ["Accuracy", "Precision", "Sensitivity_Recall", "Specificity", "F1_Score", "ROC_AUC"]
    plot_df = summary_df.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1.05)
    plt.title("Steganalysis Model Performance Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_barchart.png"), dpi=300)
    plt.close()
    
    # Save a sample residual visualization
    sample_clean = os.path.join(CLEAN_DIR, sorted(os.listdir(CLEAN_DIR))[0])
    sample_stego = os.path.join(STEGO_DIR, sorted(os.listdir(STEGO_DIR))[0])
    
    _, g_clean = load_and_preprocess(sample_clean)
    _, g_stego = load_and_preprocess(sample_stego)
    _, r_clean = compute_residual(g_clean)
    _, r_stego = compute_residual(g_stego)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(g_clean, cmap="gray")
    axes[0, 0].set_title("Clean Cover Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(r_clean, cmap="hot")
    axes[0, 1].set_title("Clean Noise Residual R")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(g_stego, cmap="gray")
    axes[1, 0].set_title("Stego Image")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(r_stego, cmap="hot")
    axes[1, 1].set_title("Stego Noise Residual R")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residual_visualization.png"), dpi=300)
    plt.close()
    print("Saved residual visualization to results/residual_visualization.png")

if __name__ == "__main__":
    train_and_evaluate_models()
