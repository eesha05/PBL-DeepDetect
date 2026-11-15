"""
DeepDetect - Phase 3: CNN Training and Evaluation
-------------------------------------------------
Trains the Residual Steganalysis CNN using the shared cnn_preprocessing module.
Features:
  - Input: 224x224x3 noise residual maps from cnn_preprocessing.py
  - Strictly held-out test split by scene/group ID (25% scenes = 50 images, zero leakage)
  - Train/Validation split by group ID (80/20 of remaining scenes)
  - Architecture: Multi-scale residual conv blocks + BatchNorm + LeakyReLU + GAP + Dense
  - Callbacks: ModelCheckpoint (best val_loss/val_accuracy) + ReduceLROnPlateau + EarlyStopping
  - Strictly held-out test set evaluation with accuracy, precision, recall, F1, ROC-AUC, FPR, FNR, confusion matrix
  - Saves: models/cnn_model.keras, results/X_test.npy, results/y_test.npy, results/cnn_training_curves.png, etc.
"""

import os
import sys

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

from cnn_preprocessing import preprocess_image_for_cnn, IMG_SIZE

CLEAN_DIR = os.path.join("data", "clean")
STEGO_DIR = os.path.join("data", "stego")
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_dataset():
    """
    Loads clean and stego images using unified cnn_preprocessing.py.
    Extracts group_id from filename digits to enforce scene-level grouping.
    """
    images = []
    labels = []
    group_ids = []
    filenames = []

    clean_files = sorted([f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    stego_files = sorted([f for f in os.listdir(STEGO_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for f in clean_files:
        path = os.path.join(CLEAN_DIR, f)
        tensor = preprocess_image_for_cnn(path, target_size=IMG_SIZE)
        images.append(tensor)
        labels.append(0)
        base = os.path.splitext(f)[0]
        digits = "".join([c for c in base if c.isdigit()])
        group_ids.append(digits if digits else base)
        filenames.append(f)

    for f in stego_files:
        path = os.path.join(STEGO_DIR, f)
        tensor = preprocess_image_for_cnn(path, target_size=IMG_SIZE)
        images.append(tensor)
        labels.append(1)
        base = os.path.splitext(f)[0]
        digits = "".join([c for c in base if c.isdigit()])
        group_ids.append(digits if digits else base)
        filenames.append(f)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), np.array(group_ids), filenames

def build_residual_steganalysis_cnn(input_shape=(224, 224, 3)):
    """
    Steganalysis CNN Architecture tailored for spatial noise residuals:
    - Multi-scale convolutional feature extractors
    - Preserves sign information via LeakyReLU (residual negative values carry stego noise)
    - Skip-connection residual block to maintain subtle gradient flow
    - Global Average Pooling (GAP) eliminates large Flatten layer, combating overfitting
    - Binary classification sigmoid head
    """
    inputs = layers.Input(shape=input_shape, name="residual_input")

    # Stage 1: Noise feature extraction
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 112x112

    # Stage 2: Spatial noise correlations
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 56x56
    x = layers.Dropout(0.2)(x)

    # Stage 3: Residual bottleneck block
    shortcut = x
    y = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(negative_slope=0.1)(y)
    y = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    x = layers.Add()([shortcut, y])
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 28x28
    x = layers.Dropout(0.25)(x)

    # Stage 4: High-order statistical aggregation
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 14x14
    x = layers.Dropout(0.3)(x)

    # Global Average Pooling
    gap = layers.GlobalAveragePooling2D()(x)

    # Regularized Dense classification head
    d = layers.Dense(64, use_bias=False)(gap)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(negative_slope=0.1)(d)
    d = layers.Dropout(0.4)(d)

    outputs = layers.Dense(1, activation="sigmoid", name="stego_prob")(d)

    model = models.Model(inputs=inputs, outputs=outputs, name="DeepDetect_Residual_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_and_evaluate_cnn():
    print("=" * 60)
    print("PHASE 3: RESIDUAL CNN TRAINING & EVALUATION")
    print("=" * 60)

    print("Loading dataset via cnn_preprocessing (224x224x3)...")
    X, y, groups, filenames = load_dataset()
    print(f"Total dataset: {len(X)} images (Clean: {np.sum(y==0)}, Stego: {np.sum(y==1)})")
    print(f"Input shape: {X.shape[1:]}")

    # 1. Held-out Test Split (25% scenes = 25 clean + 25 stego = 50 images)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_val_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]
    groups_train_val = groups[train_val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]
    filenames_test = [filenames[i] for i in test_idx]

    # Save held-out test data for downstream tasks
    np.save(os.path.join(RESULTS_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)
    test_meta = pd.DataFrame({
        "filename": filenames_test,
        "true_label": y_test,
        "group_id": groups[test_idx]
    })
    test_meta.to_csv(os.path.join(RESULTS_DIR, "cnn_test_set_metadata.csv"), index=False)

    # 2. Train / Validation Split (80% train, 20% val from remaining scenes)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups=groups_train_val))

    X_train, y_train = X_train_val[tr_idx], y_train_val[tr_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

    print(f"Dataset Partitioning: Train={len(X_train)} | Val={len(X_val)} | Held-out Test={len(X_test)}")
    print(f"Held-out Test balance: Clean={np.sum(y_test==0)}, Stego={np.sum(y_test==1)}")

    # 3. Model construction
    model = build_residual_steganalysis_cnn(input_shape=X.shape[1:])
    model.summary()

    model_path = os.path.join(MODELS_DIR, "cnn_model.keras")

    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    # 4. Training
    print("\nStarting CNN Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=cbs,
        verbose=1
    )

    # Ensure best weights are loaded for evaluation
    if os.path.exists(model_path):
        print(f"Loading best checkpoint from {model_path} for final evaluation...")
        model = models.load_model(model_path)

    # 5. Training Curves Plot
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss", marker="o", markersize=3)
    plt.plot(history.history["val_loss"], label="Val Loss", marker="s", markersize=3)
    plt.title("Residual CNN Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy", marker="o", markersize=3)
    plt.plot(history.history["val_accuracy"], label="Val Accuracy", marker="s", markersize=3)
    plt.title("Residual CNN Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_training_curves.png"), dpi=200)
    plt.close()

    # 6. Evaluation on Strictly Held-Out Test Set
    y_test_probs = model.predict(X_test, verbose=0).ravel()
    y_test_preds = (y_test_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_test_preds)
    prec = precision_score(y_test, y_test_preds, zero_division=0)
    rec = recall_score(y_test, y_test_preds, zero_division=0)
    f1 = f1_score(y_test, y_test_preds, zero_division=0)
    auc = roc_auc_score(y_test, y_test_probs)

    cm = confusion_matrix(y_test, y_test_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print("\n" + "=" * 60)
    print("RESIDUAL CNN HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Precision:   {prec*100:.2f}%")
    print(f"Recall:      {rec*100:.2f}% (Sensitivity)")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"FPR:         {fpr*100:.2f}%")
    print(f"FNR:         {fnr*100:.2f}%")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=" * 60)

    # Save classification report
    report_text = classification_report(y_test, y_test_preds, target_names=["Clean (0)", "Stego (1)"])
    with open(os.path.join(RESULTS_DIR, "cnn_classification_report.txt"), "w", encoding="utf-8") as rf:
        rf.write("DEEPDETECT RESIDUAL CNN CLASSIFICATION REPORT (HELD-OUT TEST SET)\n")
        rf.write("=" * 65 + "\n\n")
        rf.write(report_text)
        rf.write(f"\nROC-AUC Score: {auc:.4f}\n")
        rf.write(f"Confusion Matrix:\nTN={tn}, FP={fp}\nFN={fn}, TP={tp}\n")
        rf.write(f"FPR: {fpr*100:.2f}%\nFNR: {fnr*100:.2f}%\n")

    # Save Confusion Matrix Plot
    plt.figure(figsize=(4.5, 3.5))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Clean (0)", "Stego (1)"],
                yticklabels=["Clean (0)", "Stego (1)"])
    plt.title(f"Residual CNN Test Matrix (Acc: {acc*100:.1f}%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_cnn.png"), dpi=200)
    plt.close()

    # Save test predictions table
    cnn_test_df = pd.DataFrame({
        "Filename": filenames_test,
        "True_Label": y_test,
        "True_Class": ["Stego" if y == 1 else "Clean" for y in y_test],
        "CNN_Prob": y_test_probs,
        "CNN_Pred": y_test_preds,
        "Correct": (y_test == y_test_preds)
    })
    cnn_test_df.to_csv(os.path.join(RESULTS_DIR, "cnn_test_predictions.csv"), index=False)
    print(f"CNN model training & test evaluation complete. Saved model to {model_path}")
    print(f"Test predictions saved to {os.path.join(RESULTS_DIR, 'cnn_test_predictions.csv')}")

if __name__ == "__main__":
    train_and_evaluate_cnn()
