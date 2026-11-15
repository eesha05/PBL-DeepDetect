import os
import sys

os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(".", ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ["KERAS_BACKEND"] = "jax"

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

import keras
from keras import models

MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_quality_metrics(orig_img, perturbed_img):
    """Computes MSE and PSNR between original and perturbed image arrays (0-1 float or 0-255)."""
    # Convert to 0-255 scale for PSNR calculation
    o_255 = np.clip(orig_img * 255.0, 0, 255.0)
    p_255 = np.clip(perturbed_img * 255.0, 0, 255.0)
    mse = float(np.mean((o_255 - p_255) ** 2))
    psnr = float(10 * np.log10((255.0 ** 2) / mse)) if mse > 0 else 100.0
    return mse, psnr

def run_adversarial_robustness():
    model_path = os.path.join(MODELS_DIR, "cnn_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN model {model_path} not found.")
        
    model = models.load_model(model_path)
    print("CNN model loaded successfully.")
    
    # Load held-out test data
    X_test_path = os.path.join(RESULTS_DIR, "X_test.npy")
    y_test_path = os.path.join(RESULTS_DIR, "y_test.npy")
    meta_path = os.path.join(RESULTS_DIR, "cnn_test_set_metadata.csv")
    
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    meta_df = pd.read_csv(meta_path)
    
    print(f"Loaded held-out test set: {len(X_test)} samples ({np.sum(y_test == 0)} clean, {np.sum(y_test == 1)} stego)")
    
    # Get baseline predictions
    orig_probs = model.predict(X_test, verbose=0).ravel()
    orig_preds = (orig_probs >= 0.5).astype(int)
    
    # FGSM Perturbations
    # FGSM: x_adv = clip(x + epsilon * sign(grad_x(Loss(x, y))), 0, 1)
    # Target loss: Binary Cross Entropy
    epsilons = [0.005, 0.01, 0.02, 0.05, 0.10]
    
    # Define loss function for JAX autodiff through model
    def forward_loss(x_single, y_true_scalar):
        # x_single shape: (1, 128, 128, 3)
        prob = model(x_single, training=False)
        # binary crossentropy loss: - [y * log(p) + (1-y) * log(1-p)]
        eps = 1e-7
        prob_clipped = jnp.clip(prob, eps, 1.0 - eps)
        loss = - (y_true_scalar * jnp.log(prob_clipped) + (1.0 - y_true_scalar) * jnp.log(1.0 - prob_clipped))
        return jnp.squeeze(loss)

    grad_fn = jax.grad(forward_loss, argnums=0)
    
    robustness_records = []
    
    # Compute gradients for each test sample
    print("Computing FGSM adversarial gradients...")
    gradients = []
    for i in range(len(X_test)):
        x_in = jnp.expand_dims(X_test[i], axis=0)
        y_val = float(y_test[i])
        g = grad_fn(x_in, y_val)
        sign_g = np.sign(np.array(g))
        gradients.append(sign_g[0])
    gradients = np.array(gradients)
    
    # Evaluate across epsilon strengths
    for eps in epsilons:
        X_perturbed = np.clip(X_test + eps * gradients, 0.0, 1.0)
        pert_probs = model.predict(X_perturbed, verbose=0).ravel()
        pert_preds = (pert_probs >= 0.5).astype(int)
        
        for i in range(len(X_test)):
            true_lbl = int(y_test[i])
            o_prob = float(orig_probs[i])
            o_pred = int(orig_preds[i])
            p_prob = float(pert_probs[i])
            p_pred = int(pert_preds[i])
            
            mse_val, psnr_val = compute_quality_metrics(X_test[i], X_perturbed[i])
            
            prediction_changed = (o_pred != p_pred)
            
            # Evasion definition: Stego image (true label 1) originally predicted stego (1),
            # but after FGSM perturbation shifts prediction to clean (0).
            is_successful_evasion = (true_lbl == 1 and o_pred == 1 and p_pred == 0)
            
            # False alarm shift: Clean image (true label 0) originally clean (0),
            # but perturbed into false alarm (1).
            is_false_alarm_shift = (true_lbl == 0 and o_pred == 0 and p_pred == 1)
            
            robustness_records.append({
                "Epsilon": eps,
                "Sample_Index": i,
                "Filename": meta_df.iloc[i]["filename"],
                "True_Label": true_lbl,
                "Label_Name": "Stego" if true_lbl == 1 else "Clean",
                "Original_Prob": o_prob,
                "Original_Pred": o_pred,
                "Perturbed_Prob": p_prob,
                "Perturbed_Pred": p_pred,
                "Prediction_Changed": prediction_changed,
                "Evasion_Succeeded": is_successful_evasion,
                "False_Alarm_Shift": is_false_alarm_shift,
                "MSE": mse_val,
                "PSNR_dB": psnr_val
            })
            
    df_rob = pd.DataFrame(robustness_records)
    out_csv = os.path.join(RESULTS_DIR, "cnn_robustness_results.csv")
    df_rob.to_csv(out_csv, index=False)
    print(f"\nSaved detailed FGSM robustness results to {out_csv}")
    
    # Aggregated summary per epsilon
    agg_summary = []
    for eps in epsilons:
        sub = df_rob[df_rob["Epsilon"] == eps]
        total_samples = len(sub)
        pred_changes = sub["Prediction_Changed"].sum()
        pct_change = (pred_changes / total_samples) * 100.0
        
        stego_sub = sub[sub["True_Label"] == 1]
        evasions = stego_sub["Evasion_Succeeded"].sum()
        clean_sub = sub[sub["True_Label"] == 0]
        false_alarms = clean_sub["False_Alarm_Shift"].sum()
        
        mean_psnr = sub["PSNR_dB"].mean()
        mean_mse = sub["MSE"].mean()
        
        agg_summary.append({
            "Epsilon": eps,
            "Total_Samples": total_samples,
            "Prediction_Changes": int(pred_changes),
            "Prediction_Change_Pct": pct_change,
            "Stego_Evasions": int(evasions),
            "Clean_False_Alarms": int(false_alarms),
            "Mean_PSNR_dB": float(mean_psnr),
            "Mean_MSE": float(mean_mse)
        })
        
    df_agg = pd.DataFrame(agg_summary)
    agg_csv = os.path.join(RESULTS_DIR, "cnn_adversarial_summary.csv")
    df_agg.to_csv(agg_csv, index=False)
    print("\n--- FGSM ADVERSARIAL EVALUATION SUMMARY ---")
    print(df_agg)
    
    # Plot: FGSM Epsilon vs Prediction Changes & PSNR
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    color = 'tab:red'
    ax1.set_xlabel('FGSM Epsilon (Perturbation Magnitude)')
    ax1.set_ylabel('Prediction Change (%)', color=color)
    ax1.plot(df_agg["Epsilon"], df_agg["Prediction_Change_Pct"], marker='o', color=color, linewidth=2, label="Prediction Changes (%)")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Mean PSNR (dB)', color=color)
    ax2.plot(df_agg["Epsilon"], df_agg["Mean_PSNR_dB"], marker='s', color=color, linewidth=2, linestyle='--', label="Mean PSNR (dB)")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("CNN Adversarial Robustness: Prediction Shift vs Image Quality (PSNR)")
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_adversarial_tradeoff.png"), dpi=200)
    plt.close()
    print("Saved plot to results/cnn_adversarial_tradeoff.png")

if __name__ == "__main__":
    run_adversarial_robustness()
