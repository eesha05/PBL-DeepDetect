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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import keras
from keras import layers, models, callbacks

CLEAN_DIR = os.path.join("data", "clean")
STEGO_DIR = os.path.join("data", "stego")
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (128, 128)

def load_dataset():
    """Loads images, pairs them by ID to prevent data leakage, resizes and normalizes."""
    images = []
    labels = []
    group_ids = []
    filenames = []
    
    clean_files = sorted([f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    stego_files = sorted([f for f in os.listdir(STEGO_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for f in clean_files:
        path = os.path.join(CLEAN_DIR, f)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            images.append(img.astype(np.float32) / 255.0)
            labels.append(0)
            base = os.path.splitext(f)[0]
            digits = "".join([c for c in base if c.isdigit()])
            group_ids.append(digits if digits else base)
            filenames.append(f)
            
    for f in stego_files:
        path = os.path.join(STEGO_DIR, f)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            images.append(img.astype(np.float32) / 255.0)
            labels.append(1)
            base = os.path.splitext(f)[0]
            digits = "".join([c for c in base if c.isdigit()])
            group_ids.append(digits if digits else base)
            filenames.append(f)
            
    return np.array(images), np.array(labels), np.array(group_ids), filenames

def build_cnn(input_shape=(128, 128, 3)):
    """DeepDetect Steganalysis CNN Model."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense Classifier
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_cnn():
    X, y, groups, filenames = load_dataset()
    print(f"Loaded {len(X)} images. Classes: {np.bincount(y)}")
    
    # 1. Group split: Separate Held-out Test Set (25% scenes)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train_val, y_train_val, groups_train_val = X[train_val_idx], y[train_val_idx], groups[train_val_idx]
    X_test, y_test, filenames_test = X[test_idx], y[test_idx], [filenames[i] for i in test_idx]
    
    # Save held-out test set indices and metadata
    test_meta = pd.DataFrame({
        "filename": filenames_test,
        "true_label": y_test,
        "group_id": groups[test_idx]
    })
    test_meta.to_csv(os.path.join(RESULTS_DIR, "cnn_test_set_metadata.csv"), index=False)
    np.save(os.path.join(RESULTS_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)
    
    # 2. Group split: Train (80%) and Validation (20%) from remaining scenes
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups=groups_train_val))
    
    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Held-out Test samples: {len(X_test)}")
    
    model = build_cnn(input_shape=(128, 128, 3))
    model.summary()
    
    model_save_path = os.path.join(MODELS_DIR, "cnn_model.keras")
    
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = callbacks.ModelCheckpoint(
        model_save_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=35,
        batch_size=16,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Evaluate on held-out test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n--- CNN TEST SET PERFORMANCE ---")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    # Save training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss", marker='o')
    plt.plot(history.history["val_loss"], label="Val Loss", marker='s')
    plt.title("CNN Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy", marker='o')
    plt.plot(history.history["val_accuracy"], label="Val Accuracy", marker='s')
    plt.title("CNN Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_training_curves.png"), dpi=200)
    plt.close()
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Clean (0)", "Stego (1)"],
                yticklabels=["Clean (0)", "Stego (1)"])
    plt.title("CNN Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_cnn.png"), dpi=200)
    plt.close()
    
    print(f"CNN model saved successfully to {model_save_path}")

if __name__ == "__main__":
    train_cnn()
