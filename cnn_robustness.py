"""
DeepDetect - Phase 4: Adversarial and Robustness Evaluation
-----------------------------------------------------------
Evaluates the Steganalysis CNN against:
  1. FGSM (Fast Gradient Sign Method)
  2. PGD (Projected Gradient Descent, 7-step iterative)
  3. Controlled Gaussian Noise

Correctly accounts for:
  - Input domain: Residual feature maps in range [-1.0, 1.0] (NOT clipped to [0, 1])
  - Perturbations projected within [-1.0, 1.0]
  - Clear distinction between feature-space perturbation metrics (L2/Linf norm)
    and image-space metrics.
  - Evaluation on strictly held-out test set (50 samples: 25 clean, 25 stego).
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
import jax
import jax.numpy as jnp

import keras
from keras import models

MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_feature_perturbation_metrics(orig_tensor, pert_tensor):
    """
    Computes feature-space perturbation distance:
      - MSE: Mean squared difference
      - Linf: Maximum absolute difference
      - L2: Euclidean distance
      - PSNR_dB: Normalized signal-to-noise ratio in feature space [-1, 1] range (peak-to-peak = 2.0)
    """
    diff = pert_tensor - orig_tensor
    mse = float(np.mean(diff ** 2))
    linf = float(np.max(np.abs(diff)))
    l2 = float(np.sqrt(np.sum(diff ** 2)))
    # Range is 2.0 (-1 to +1)
    psnr_db = float(10 * np.log10((2.0 ** 2) / mse)) if mse > 0 else 100.0
    return mse, linf, l2, psnr_db

def run_adversarial_robustness_suite():
    print("=" * 60)
    print("PHASE 4: CNN ADVERSARIAL ROBUSTNESS EVALUATION")
    print("=" * 60)

    model_path = os.path.join(MODELS_DIR, "cnn_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained CNN model not found at {model_path}")

    model = models.load_model(model_path)
    print("Residual CNN model loaded successfully.")

    # Load strictly held-out test set
    X_test = np.load(os.path.join(RESULTS_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(RESULTS_DIR, "y_test.npy"))
    meta_df = pd.read_csv(os.path.join(RESULTS_DIR, "cnn_test_set_metadata.csv"))

    n_samples = len(X_test)
    n_clean = int(np.sum(y_test == 0))
    n_stego = int(np.sum(y_test == 1))
    print(f"Loaded held-out test set: {n_samples} samples ({n_clean} clean, {n_stego} stego)")
    print(f"Tensor value bounds: [{np.min(X_test):.3f}, {np.max(X_test):.3f}]")

    # Baseline clean test predictions
    orig_probs = model.predict(X_test, verbose=0).ravel()
    orig_preds = (orig_probs >= 0.5).astype(int)
    base_acc = float(np.mean(orig_preds == y_test))
    print(f"Baseline unattacked held-out accuracy: {base_acc*100:.2f}%")

    # JAX autodiff loss function
    def forward_bce_loss(x_single, y_true_scalar):
        prob = model(x_single, training=False)
        eps = 1e-7
        p = jnp.clip(prob, eps, 1.0 - eps)
        loss = - (y_true_scalar * jnp.log(p) + (1.0 - y_true_scalar) * jnp.log(1.0 - p))
        return jnp.squeeze(loss)

    grad_fn = jax.grad(forward_bce_loss, argnums=0)

    epsilons = [0.005, 0.01, 0.02, 0.05, 0.10]

    # 1. FGSM Gradients Computation
    print("\n[1/3] Computing FGSM (Fast Gradient Sign Method) perturbations...")
    fgsm_gradients = []
    for i in range(n_samples):
        x_in = jnp.expand_dims(X_test[i], axis=0)
        y_val = float(y_test[i])
        g = grad_fn(x_in, y_val)
        fgsm_gradients.append(np.sign(np.array(g))[0])
    fgsm_gradients = np.array(fgsm_gradients, dtype=np.float32)

    # 2. PGD Multi-Step Iterative Perturbations
    print("[2/3] Computing PGD (Projected Gradient Descent, 7 steps)...")
    pgd_perturbed_sets = {}
    pgd_steps = 7
    for eps in epsilons:
        alpha = eps / 4.0
        rng = np.random.default_rng(42)
        random_noise = rng.uniform(-eps, eps, X_test.shape).astype(np.float32)
        # Proper range is [-1.0, 1.0]
        x_adv = np.clip(X_test + random_noise, -1.0, 1.0)

        for step in range(pgd_steps):
            step_grads = []
            for i in range(n_samples):
                g = grad_fn(jnp.expand_dims(x_adv[i], axis=0), float(y_test[i]))
                step_grads.append(np.sign(np.array(g))[0])
            step_grads = np.array(step_grads, dtype=np.float32)

            x_adv = x_adv + alpha * step_grads
            eta = np.clip(x_adv - X_test, -eps, eps)
            x_adv = np.clip(X_test + eta, -1.0, 1.0)

        pgd_perturbed_sets[eps] = x_adv

    # 3. Gaussian Noise Perturbations
    print("[3/3] Evaluating Gaussian Noise robustness...")
    noise_sigmas = [0.01, 0.02, 0.05, 0.10]

    detailed_records = []
    summary_records = []

    def evaluate_attack(attack_name, eps_or_param, x_perturbed):
        pert_probs = model.predict(x_perturbed, verbose=0).ravel()
        pert_preds = (pert_probs >= 0.5).astype(int)

        acc = float(np.mean(pert_preds == y_test))
        pred_changes = int(np.sum(orig_preds != pert_preds))
        pred_change_pct = (pred_changes / n_samples) * 100.0

        stego_mask = (y_test == 1)
        stego_evasions = int(np.sum((orig_preds[stego_mask] == 1) & (pert_preds[stego_mask] == 0)))
        stego_evasion_pct = (stego_evasions / n_stego) * 100.0 if n_stego > 0 else 0.0

        clean_mask = (y_test == 0)
        clean_false_alarms = int(np.sum((orig_preds[clean_mask] == 0) & (pert_preds[clean_mask] == 1)))
        clean_false_alarm_pct = (clean_false_alarms / n_clean) * 100.0 if n_clean > 0 else 0.0

        mses, linfs, psnrs = [], [], []
        for i in range(n_samples):
            m, lf, l2, p = compute_feature_perturbation_metrics(X_test[i], x_perturbed[i])
            mses.append(m)
            linfs.append(lf)
            psnrs.append(p)

            detailed_records.append({
                "Attack_Type": attack_name,
                "Perturbation_Strength": eps_or_param,
                "Sample_Index": i,
                "Filename": meta_df.iloc[i]["filename"],
                "True_Label": int(y_test[i]),
                "Label_Name": "Stego" if y_test[i] == 1 else "Clean",
                "Original_Prob": float(orig_probs[i]),
                "Original_Pred": int(orig_preds[i]),
                "Perturbed_Prob": float(pert_probs[i]),
                "Perturbed_Pred": int(pert_preds[i]),
                "Prediction_Changed": bool(orig_preds[i] != pert_preds[i]),
                "Evasion_Succeeded": bool(y_test[i] == 1 and orig_preds[i] == 1 and pert_preds[i] == 0),
                "False_Alarm_Shift": bool(y_test[i] == 0 and orig_preds[i] == 0 and pert_preds[i] == 1),
                "Feature_MSE": m,
                "Feature_Linf": lf,
                "Feature_PSNR_dB": p
            })

        summary_records.append({
            "Attack_Type": attack_name,
            "Perturbation_Param": eps_or_param,
            "Accuracy": acc,
            "Accuracy_Drop": base_acc - acc,
            "Prediction_Changes": pred_changes,
            "Prediction_Change_Pct": pred_change_pct,
            "Stego_Evasions": stego_evasions,
            "Stego_Evasion_Pct": stego_evasion_pct,
            "Clean_False_Alarms": clean_false_alarms,
            "Clean_False_Alarm_Pct": clean_false_alarm_pct,
            "Mean_Feature_PSNR_dB": float(np.mean(psnrs)),
            "Mean_Feature_MSE": float(np.mean(mses)),
            "Max_Feature_Linf": float(np.max(linfs))
        })

    # Evaluate FGSM across epsilons
    for eps in epsilons:
        x_fgsm = np.clip(X_test + eps * fgsm_gradients, -1.0, 1.0)
        evaluate_attack("FGSM", eps, x_fgsm)

    # Evaluate PGD across epsilons
    for eps in epsilons:
        evaluate_attack("PGD", eps, pgd_perturbed_sets[eps])

    # Evaluate Gaussian Noise
    for sig in noise_sigmas:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, sig, X_test.shape).astype(np.float32)
        x_noisy = np.clip(X_test + noise, -1.0, 1.0)
        evaluate_attack("Gaussian_Noise", sig, x_noisy)

    # Save results
    df_det = pd.DataFrame(detailed_records)
    df_sum = pd.DataFrame(summary_records)

    det_csv = os.path.join(RESULTS_DIR, "cnn_robustness_results.csv")
    sum_csv = os.path.join(RESULTS_DIR, "cnn_adversarial_summary.csv")
    df_det.to_csv(det_csv, index=False)
    df_sum.to_csv(sum_csv, index=False)

    print("\n--- ADVERSARIAL ROBUSTNESS SUMMARY TABLE ---")
    print(df_sum[["Attack_Type", "Perturbation_Param", "Accuracy", "Prediction_Change_Pct", "Stego_Evasion_Pct", "Clean_False_Alarm_Pct", "Mean_Feature_PSNR_dB"]])

    # Plot Tradeoff Curves
    plt.figure(figsize=(11, 4.8))

    plt.subplot(1, 2, 1)
    fgsm_sub = df_sum[df_sum["Attack_Type"] == "FGSM"]
    pgd_sub = df_sum[df_sum["Attack_Type"] == "PGD"]
    plt.plot(fgsm_sub["Perturbation_Param"], fgsm_sub["Accuracy"] * 100, marker='o', linewidth=2, label="FGSM (Single-Step)")
    plt.plot(pgd_sub["Perturbation_Param"], pgd_sub["Accuracy"] * 100, marker='s', linewidth=2, label="PGD (7-Step Iterative)")
    plt.axhline(y=base_acc * 100, color='gray', linestyle='--', label=f"Clean Test Baseline ({base_acc*100:.1f}%)")
    plt.title("Model Accuracy Degradation Under Attack")
    plt.xlabel("Perturbation Budget ε (L_inf)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fgsm_sub["Perturbation_Param"], fgsm_sub["Stego_Evasion_Pct"], marker='o', linewidth=2, label="FGSM Stego Evasion")
    plt.plot(pgd_sub["Perturbation_Param"], pgd_sub["Stego_Evasion_Pct"], marker='s', linewidth=2, label="PGD Stego Evasion")
    plt.title("Stego Sample Evasion Rate (% Stego -> Clean)")
    plt.xlabel("Perturbation Budget ε (L_inf)")
    plt.ylabel("Evasion Rate (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_adversarial_tradeoff.png"), dpi=200)
    plt.close()

    print(f"\nDetailed sample records saved -> {det_csv}")
    print(f"Summary table saved -> {sum_csv}")
    print("Tradeoff visualization saved -> results/cnn_adversarial_tradeoff.png")

if __name__ == "__main__":
    run_adversarial_robustness_suite()
