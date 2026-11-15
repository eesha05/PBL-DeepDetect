import os
import sys

# --- Must be set BEFORE any matplotlib / keras import ---
os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(".", ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ["KERAS_BACKEND"] = "jax"

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

import keras
from keras import layers, models, callbacks

# ─────────────────────────── PATHS ───────────────────────────
CLEAN_DIR   = os.path.join("data", "clean")
STEGO_DIR   = os.path.join("data", "stego")
MODELS_DIR  = "models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────── KEY SETTINGS ────────────────────
# 224×224 is crucial — at 128×128, bilinear downsampling from 512×512
# destroys ~93 % of LSB-modified pixels.  At 224×224 we keep ~75 %.
IMG_SIZE = (224, 224)


# ═══════════════════════════════════════════════════════════════
#  RESIDUAL PRE-PROCESSING
#  Instead of raw RGB pixels (which look identical for clean vs stego)
#  we compute a 3-channel HIGH-PASS NOISE MAP that amplifies the
#  subtle LSB perturbations and suppresses the visible image content.
# ═══════════════════════════════════════════════════════════════

# SRM / KV-style 5×5 high-pass kernel (Fridrich & Kodovsky 2012)
_KV_KERNEL = np.array([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8, -12, 8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1]
], dtype=np.float32) / 12.0


def compute_residual_input(bgr_image):
    """
    Convert a BGR image (OpenCV format) into a 3-channel residual map
    suitable as CNN input.

    Channel layout:
        0 – Gaussian residual:  Gray - GaussianBlur(Gray)
        1 – KV high-pass:       filter2D(Gray, kv_kernel) / 4
        2 – Laplacian magnitude: |Laplacian(Gray)| / 8

    All channels are clipped to [-1, 1] and kept as float32.
    The residuals amplify the ~1-bit LSB difference that is otherwise
    invisible in the raw RGB domain, allowing the CNN to learn from it.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Channel 0: Gaussian residual (σ=1.0, 5×5 window)
    blurred  = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)
    gauss_r  = (gray - blurred) / 4.0          # scale to ~[-1, 1]

    # Channel 1: KV / SRM high-pass residual
    kv_r     = cv2.filter2D(gray, -1, _KV_KERNEL) / 4.0

    # Channel 2: Laplacian edge-noise residual
    lap      = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_r    = lap / 8.0

    stacked  = np.stack([gauss_r, kv_r, lap_r], axis=-1)
    stacked  = np.clip(stacked, -1.0, 1.0)
    return stacked.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  DATASET LOADING
# ═══════════════════════════════════════════════════════════════

def load_dataset():
    """
    Loads images, pairs them by scene ID to prevent data leakage,
    resizes to IMG_SIZE, and converts each image to a residual map.
    Returns X (N, H, W, 3 residual), y (N,), groups (N,), filenames.
    """
    images   = []
    labels   = []
    group_ids = []
    filenames = []

    clean_files = sorted(
        [f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )
    stego_files = sorted(
        [f for f in os.listdir(STEGO_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )

    def _load_one(path, label):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return None
        img_bgr = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        residual = compute_residual_input(img_bgr)
        return residual, label

    for f in clean_files:
        result = _load_one(os.path.join(CLEAN_DIR, f), 0)
        if result is None:
            continue
        residual, label = result
        images.append(residual)
        labels.append(label)
        base   = os.path.splitext(f)[0]
        digits = "".join([c for c in base if c.isdigit()])
        group_ids.append(digits if digits else base)
        filenames.append(f)

    for f in stego_files:
        result = _load_one(os.path.join(STEGO_DIR, f), 1)
        if result is None:
            continue
        residual, label = result
        images.append(residual)
        labels.append(label)
        base   = os.path.splitext(f)[0]
        digits = "".join([c for c in base if c.isdigit()])
        group_ids.append(digits if digits else base)
        filenames.append(f)

    return np.array(images), np.array(labels), np.array(group_ids), filenames


# ═══════════════════════════════════════════════════════════════
#  CNN ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

def build_residual_steganalysis_cnn(input_shape=(224, 224, 3)):
    """
    Steganalysis CNN that operates on 3-channel noise residual maps.

    Design rationale
    ----------------
    * Input = pre-computed residuals (Gaussian + KV + Laplacian),
      NOT raw pixels.  This amplifies the sub-visual LSB signal.
    * Small 3×3 kernels capture local pixel-difference correlations.
    * BatchNormalization + LeakyReLU (not ReLU) preserve negative
      residual values that carry steganographic information.
    * GlobalAveragePooling avoids a large Flatten layer and reduces
      overfitting on the small 100-pair dataset.
    * Moderate Dropout (0.3–0.4) regularizes the classification head.
    """
    inputs = layers.Input(shape=input_shape, name="residual_input")

    # ── Block 1: Noise capture (3×3 kernel, preserve residual polarity)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)      # 112×112

    # ── Block 2: Feature detection
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)      # 56×56
    x = layers.Dropout(0.2)(x)

    # ── Block 3: Residual-skip block (retains weak signal through shortcut)
    shortcut = x
    y = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(negative_slope=0.1)(y)
    y = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    x = layers.Add()([shortcut, y])
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)      # 28×28
    x = layers.Dropout(0.25)(x)

    # ── Block 4: Higher-level aggregation
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)      # 14×14
    x = layers.Dropout(0.3)(x)

    # ── Global Average Pooling → compact 128-dim descriptor
    gap = layers.GlobalAveragePooling2D()(x)

    # ── Classification head
    d   = layers.Dense(64, use_bias=False)(gap)
    d   = layers.BatchNormalization()(d)
    d   = layers.LeakyReLU(negative_slope=0.1)(d)
    d   = layers.Dropout(0.4)(d)
    out = layers.Dense(1, activation="sigmoid", name="stego_prob")(d)

    model = models.Model(inputs=inputs, outputs=out,
                         name="DeepDetect_Residual_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ═══════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════

def train_and_evaluate_cnn():
    print("Loading dataset as residual maps (224×224) …")
    X, y, groups, filenames = load_dataset()
    print(f"Loaded {len(X)} images  ({np.sum(y==0)} clean, {np.sum(y==1)} stego)")
    print(f"Residual input shape: {X.shape[1:]}")

    # ── 1. Held-out test split (25 % of scenes, no leakage across pairs)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_val_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train_val  = X[train_val_idx]
    y_train_val  = y[train_val_idx]
    groups_tv    = groups[train_val_idx]
    X_test       = X[test_idx]
    y_test       = y[test_idx]
    filenames_test = [filenames[i] for i in test_idx]

    # Save held-out test set for cnn_robustness.py (now as residual maps)
    np.save(os.path.join(RESULTS_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)
    pd.DataFrame({
        "filename":   filenames_test,
        "true_label": y_test,
        "group_id":   groups[test_idx]
    }).to_csv(os.path.join(RESULTS_DIR, "cnn_test_set_metadata.csv"), index=False)

    # ── 2. Train-validation split (80 / 20 from remaining scenes)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups=groups_tv))

    X_train, y_train = X_train_val[tr_idx],  y_train_val[tr_idx]
    X_val,   y_val   = X_train_val[val_idx], y_train_val[val_idx]

    print(f"\nSplit — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── 3. Build & summarise model
    model = build_residual_steganalysis_cnn(input_shape=X.shape[1:])
    print("\n--- RESIDUAL CNN ARCHITECTURE ---")
    model.summary()

    model_path = os.path.join(MODELS_DIR, "cnn_model.keras")

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    # ── 4. Train
    print("\nStarting CNN training (residual domain, 224×224) …")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=16,
        callbacks=cbs,
        verbose=1
    )

    # ── 5. Training curves
    plt.figure(figsize=(12, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"],     label="Train Loss",  marker="o", markersize=3)
    plt.plot(history.history["val_loss"], label="Val Loss",    marker="s", markersize=3)
    plt.title("CNN Training & Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Binary Cross-Entropy")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"],     label="Train Accuracy", marker="o", markersize=3)
    plt.plot(history.history["val_accuracy"], label="Val Accuracy",   marker="s", markersize=3)
    plt.title("CNN Training & Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_training_curves.png"), dpi=200)
    plt.close()

    # ── 6. Held-out test evaluation
    y_probs = model.predict(X_test, verbose=0).ravel()
    y_preds = (y_probs >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_preds)
    prec = precision_score(y_test, y_preds, zero_division=0)
    rec  = recall_score(y_test, y_preds, zero_division=0)
    f1   = f1_score(y_test, y_preds, zero_division=0)
    auc  = roc_auc_score(y_test, y_probs)
    cm   = confusion_matrix(y_test, y_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print("\n================ CNN HELD-OUT TEST EVALUATION ================")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Precision:   {prec*100:.2f}%")
    print(f"Recall:      {rec*100:.2f}%  (Sensitivity)")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"FPR:         {fpr*100:.2f}%")
    print(f"FNR:         {fnr*100:.2f}%")
    print("==============================================================")

    # Save classification report
    report_text = classification_report(y_test, y_preds,
                                        target_names=["Clean (0)", "Stego (1)"])
    with open(os.path.join(RESULTS_DIR, "cnn_classification_report.txt"),
              "w", encoding="utf-8") as fh:
        fh.write("DEEPDETECT RESIDUAL CNN CLASSIFICATION REPORT (HELD-OUT TEST SET)\n")
        fh.write("=" * 68 + "\n\n")
        fh.write(report_text)
        fh.write(f"\nROC-AUC Score: {auc:.4f}\n")
        fh.write(f"Confusion Matrix:\nTN={tn}, FP={fp}\nFN={fn}, TP={tp}\n")

    # Confusion matrix plot
    import seaborn as sns
    plt.figure(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Clean", "Stego"],
                yticklabels=["Clean", "Stego"])
    plt.title(f"Residual CNN Test (Acc: {acc*100:.1f}%)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_cnn.png"), dpi=200)
    plt.close()

    # Prediction CSV
    pd.DataFrame({
        "Filename":   filenames_test,
        "True_Label": y_test,
        "True_Class": ["Stego" if v == 1 else "Clean" for v in y_test],
        "CNN_Prob":   y_probs,
        "CNN_Pred":   y_preds,
        "Correct":    (y_test == y_preds)
    }).to_csv(os.path.join(RESULTS_DIR, "cnn_test_predictions.csv"), index=False)

    print(f"\nCNN training complete. Model saved → {model_path}")
    print(f"X_test.npy updated (residual maps, shape {X_test.shape})")


if __name__ == "__main__":
    train_and_evaluate_cnn()
