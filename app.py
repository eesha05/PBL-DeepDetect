import os
import sys

# Ensure local matplotlib configdir to prevent Windows permission issues
os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(".", ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ["KERAS_BACKEND"] = "jax"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from skimage.metrics import structural_similarity as compute_ssim

# DeepDetect modules
from residual_analysis import (
    compute_residual, extract_edge_features, extract_residual_features,
    extract_all_features, MODELS_DIR, RESULTS_DIR
)
from deepdetect_pipeline import sobel_edges, laplacian_edges, canny_edges
from payload_robustness import calculate_mse, calculate_psnr, calculate_ber, embed_lsb, extract_lsb
from cnn_preprocessing import preprocess_image_for_cnn
st.set_page_config(
    page_title="DeepDetect — AI Steganalysis & Robustness",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.1rem; }
    .sub-title { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.2rem; }
    .stAlert { border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🛡️ DeepDetect System")
st.sidebar.markdown("**AI Image Steganalysis & Robustness Benchmark**")

sections = [
    "A. Image Analysis & Feature Extraction",
    "B. AI-Based Detection & Live Inference",
    "C. CNN Performance & Evaluation",
    "D. Adversarial Robustness (FGSM / PGD / Noise)",
    "E. Payload Capacity & Recovery Analysis",
    "F. Core Steganography Metrics (8 Essential Values)",
    "G. Reports & Experiment Synthesis"
]

selected_section = st.sidebar.radio("Navigation / PBL Modules", sections)

# Model Loader
_CACHED_MODELS = None
def load_all_models():
    global _CACHED_MODELS
    if _CACHED_MODELS is not None:
        return _CACHED_MODELS
    models = {}
    try:
        models["edge"] = joblib.load(os.path.join(MODELS_DIR, "Edge_Only_LogisticRegression.joblib"))
        models["residual"] = joblib.load(os.path.join(MODELS_DIR, "Residual_Only_LogisticRegression.joblib"))
        models["combined"] = joblib.load(os.path.join(MODELS_DIR, "Combined_LogisticRegression.joblib"))
    except Exception as e:
        st.sidebar.error(f"Error loading LR models: {e}")
        
    try:
        import keras
        models["cnn"] = keras.models.load_model(os.path.join(MODELS_DIR, "cnn_model.keras"))
    except Exception as e:
        models["cnn"] = None
    _CACHED_MODELS = models
    return models

models_dict = load_all_models()
# ==========================================================
# UNIFIED CNN INFERENCE
# ==========================================================

def predict_cnn_probability(model, image_bgr):
    """
    Predicts the probability that an image is Stego.

    Uses exactly the same preprocessing as CNN training.
    Returns a probability between 0 and 1.
    """

    if model is None:
        return None

    if image_bgr is None:
        raise ValueError("Input image is empty or invalid.")

    # Use the same preprocessing function used during training
    processed = preprocess_image_for_cnn(image_bgr)

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    cnn_input = np.expand_dims(processed, axis=0).astype(np.float32)

    # Verify compatibility with the loaded model
    expected_shape = tuple(model.input_shape[1:])
    actual_shape = tuple(cnn_input.shape[1:])

    if expected_shape != actual_shape:
        raise ValueError(
            f"CNN input mismatch. Expected {expected_shape}, "
            f"received {actual_shape}."
        )

    prediction = model.predict(cnn_input, verbose=0)

    return float(np.asarray(prediction).reshape(-1)[0])

# ==============================================================================
# SECTION A: IMAGE ANALYSIS & FEATURE EXTRACTION
# ==============================================================================
if selected_section == "A. Image Analysis & Feature Extraction":
    st.markdown('<div class="main-title">A. Image Analysis & Feature Extraction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Spatial Decomposition & Dual-Domain Feature Extraction</div>', unsafe_allow_html=True)
    
    st.write("""
    Inspect digital images through high-frequency spatial edge representations and noise residual filtering.
    LSB steganography injects pseudorandom noise that reveals characteristic anomalies under residual computation:
    $$R = I - \\text{GaussianBlur}(I)$$
    """)
    
    uploaded_file = st.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    col_d, _ = st.columns([1, 2])
    use_sample = col_d.checkbox("Or load sample image from dataset", value=(uploaded_file is None))
    
    img_bgr = None
    if uploaded_file is not None:
        bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(bytes_data, 1)
    elif use_sample:
        demo_path = os.path.join("data", "stego", "stego_001.png")
        if os.path.exists(demo_path):
            img_bgr = cv2.imread(demo_path)
            st.caption(f"Loaded demo dataset image: `{demo_path}`")
            
    if img_bgr is not None:
        h, w, _ = img_bgr.shape
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb, (224, 224))
        gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
        
        # Decompositions
        sobel_img = sobel_edges(gray)
        laplacian_img = laplacian_edges(gray)
        canny_img = canny_edges(gray)
        res_float, res_uint8 = compute_residual(gray)
        
        st.subheader("1. Spatial Decompositions")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.image(rgb, caption=f"Original ({w}x{h})", use_column_width=True)
        c2.image(gray, caption="Grayscale", use_column_width=True)
        c3.image(sobel_img, caption="Sobel Magnitude", use_column_width=True)
        c4.image(laplacian_img, caption="Laplacian Edges", use_column_width=True)
        c5.image(canny_img, caption="Canny Edge Map", use_column_width=True)
        
        st.subheader("2. Spatial Noise Residual Image")
        cr1, cr2 = st.columns([1.5, 2.5])
        with cr1:
            st.image(res_uint8, caption="Noise Residual R (Absdiff)", use_column_width=True)
        with cr2:
            st.markdown("""
            **Noise Residual Statistics**:
            - **Residual Mean**: Measures global directional bias of subtle pixel modifications.
            - **Residual Variance & Std**: Captures additive high-frequency entropy caused by LSB flipping.
            - **Skewness & Kurtosis**: Quantifies distribution tail heaviness and non-Gaussian perturbations.
            - **10-Bin Histogram**: Encodes residual amplitude concentration.
            """)
            
        st.subheader("3. Extracted Feature Bank")
        features = extract_all_features(gray)
        feat_df = pd.DataFrame([features]).T.reset_index()
        feat_df.columns = ["Feature Name", "Calculated Value"]
        st.dataframe(feat_df)

# ==============================================================================
# SECTION B: AI-BASED DETECTION & LIVE INFERENCE
# ==============================================================================
elif selected_section == "B. AI-Based Detection & Live Inference":
    st.markdown('<div class="main-title">B. AI-Based Detection & Live Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-Model Inference: Edge LR, Residual LR, Combined LR, and Deep CNN</div>', unsafe_allow_html=True)
    
    st.write("Perform multi-model classification on any input image to determine whether it contains hidden steganographic payload.")
    
    uploaded_file = st.file_uploader("Upload Image to Detect (PNG/JPG)", type=["png", "jpg", "jpeg"], key="detect_upload")
    col_d, _ = st.columns([1, 2])
    use_sample = col_d.checkbox("Or test a dataset stego image", value=(uploaded_file is None), key="detect_sample")
    
    img_bgr = None
    if uploaded_file is not None:
        bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(bytes_data, 1)
    elif use_sample:
        demo_path = os.path.join("data", "stego", "stego_005.png")
        if os.path.exists(demo_path):
            img_bgr = cv2.imread(demo_path)
            st.caption(f"Loaded demo image: `{demo_path}`")
            
    if img_bgr is not None:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(cv2.resize(rgb, (224, 224)), cv2.COLOR_BGR2GRAY)
        
        feats = extract_all_features(gray)
        
        all_feat_df = pd.read_csv(os.path.join(RESULTS_DIR, "deepdetect_all_features.csv"))
        edge_cols = ["canny_edge_density", "sobel_mean", "sobel_std", "laplacian_mean", "laplacian_variance"]
        res_cols = [c for c in all_feat_df.columns if c.startswith("residual_") or c.startswith("mean_abs") or c.startswith("res_hist_")]
        comb_cols = edge_cols + res_cols
        
        v_edge = np.array([[feats[c] for c in edge_cols]])
        v_res = np.array([[feats[c] for c in res_cols]])
        v_comb = np.array([[feats[c] for c in comb_cols]])
        
        prob_edge = float(models_dict["edge"].predict_proba(v_edge)[0, 1])
        prob_res = float(models_dict["residual"].predict_proba(v_res)[0, 1])
        prob_comb = float(models_dict["combined"].predict_proba(v_comb)[0, 1])
        
        # CNN prediction using unified residual preprocessing
        prob_cnn = predict_cnn_probability(
            models_dict.get("cnn"),
            img_bgr)
     
            
        col_img, col_preds = st.columns([1, 1.3])
        with col_img:
            st.image(rgb, caption="Analyzed Image", use_column_width=True)
            st.metric("Primary Detector (Combined LR)", f"{prob_comb*100:.2f}% Stego Probability",
                      delta="🔴 Steganography Detected" if prob_comb >= 0.5 else "🟢 Clean / Cover Image")
            
        with col_preds:
            st.subheader("Model Decision & Confidence Matrix")
            results_table = pd.DataFrame({
                "Model Architecture": [
                    "Edge-Only Logistic Regression",
                    "Residual-Only Logistic Regression",
                    "Combined (Edge + Residual) LR",
                    "Deep Steganalysis CNN"
                ],

                "Stego Probability": [
                    f"{prob_edge * 100:.2f}%",
                    f"{prob_res * 100:.2f}%",
                    f"{prob_comb * 100:.2f}%",
                    f"{prob_cnn * 100:.2f}%" if prob_cnn is not None else "Unavailable"
                ],

                "Classification Verdict": [
                    "Stego Image" if prob_edge >= 0.5 else "Clean Image",
                    "Stego Image" if prob_res >= 0.5 else "Clean Image",
                    "Stego Image" if prob_comb >= 0.5 else "Clean Image",
                    (
                        "Stego Image" if prob_cnn >= 0.5 else "Clean Image"
                    ) if prob_cnn is not None else "CNN unavailable"
                ]
            })


            
            st.info("""
            **PBL Evaluation Note**:  
            Notice how the **Residual-Only** and **Combined LR** models accurately spot high-frequency noise deviations 
            even when the spatial image structure appears completely normal to human vision.
            """)

# ==============================================================================
# SECTION C: CNN PERFORMANCE & EVALUATION
# ==============================================================================
elif selected_section == "C. CNN Performance & Evaluation":
    st.markdown('<div class="main-title">C. CNN Performance & Comparative Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Training Convergence, Confusion Matrices, and Multi-Model ROC/PR Curves</div>', unsafe_allow_html=True)
    
    st.subheader("1. Unified Model Benchmark on Held-out Test Set")
    comp_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        st.dataframe(df_comp[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "FPR", "FNR"]])
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("2. ROC & Precision-Recall Curves")
        roc_pr_path = os.path.join(RESULTS_DIR, "roc_pr_comparison.png")
        if os.path.exists(roc_pr_path):
            st.image(roc_pr_path, caption="ROC & PR Curves (Held-out Test Split)", use_column_width=True)
    with col_c2:
        st.subheader("3. Model Performance Bar Chart")
        barchart_path = os.path.join(RESULTS_DIR, "model_comparison_barchart.png")
        if os.path.exists(barchart_path):
            st.image(barchart_path, caption="Side-by-side Metric Comparison Across Models", use_column_width=True)
            
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("4. CNN Training & Validation Curves")
        train_curves = os.path.join(RESULTS_DIR, "cnn_training_curves.png")
        if os.path.exists(train_curves):
            st.image(train_curves, caption="CNN Binary Cross-Entropy Loss & Accuracy", use_column_width=True)
    with col_t2:
        st.subheader("5. CNN Test Confusion Matrix")
        cm_cnn = os.path.join(RESULTS_DIR, "confusion_matrix_cnn.png")
        if os.path.exists(cm_cnn):
            st.image(cm_cnn, caption="CNN Classification Matrix on Test Images", use_column_width=True)
            
    rep_path = os.path.join(RESULTS_DIR, "cnn_classification_report.txt")
    if os.path.exists(rep_path):
        with open(rep_path, "r", encoding="utf-8") as f:
            st.text_area("CNN Classification Report & Diagnostics", f.read(), height=220)

# ==============================================================================
# SECTION D: ADVERSARIAL ROBUSTNESS (FGSM / PGD / NOISE)
# ==============================================================================
elif selected_section == "D. Adversarial Robustness (FGSM / PGD / Noise)":
    st.markdown('<div class="main-title">D. Adversarial Robustness & Attack Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">FGSM, Projected Gradient Descent (PGD), and Gaussian Noise Robustness</div>', unsafe_allow_html=True)
    
    st.write("""
    Adversarial attacks introduce imperceptible perturbations designed to maximize model loss:
    - **FGSM (Fast Gradient Sign Method)**: Single-step gradient ascent:
      $$x_{\\text{adv}} = \\text{clip}\\left(x + \\epsilon \\cdot \\text{sign}\\left(\\nabla_x \\mathcal{L}(x, y)\\right), 0, 1\\right)$$
    - **PGD (Projected Gradient Descent)**: Multi-step iterative gradient ascent with $L_\\infty$ ball projection:
      $$x^{t+1} = \\Pi_{x + \\mathcal{S}}\\left(x^t + \\alpha \\cdot \\text{sign}\\left(\\nabla_x \\mathcal{L}(x^t, y)\\right)\\right)$$
    """)
    
    tradeoff_img = os.path.join(RESULTS_DIR, "cnn_adversarial_tradeoff.png")
    if os.path.exists(tradeoff_img):
        st.image(tradeoff_img, caption="Adversarial Tradeoff: Attack Shift vs Image Quality Degradation", use_column_width=True)
        
    st.subheader("Adversarial & Noise Attack Summary Table")
    sum_csv = os.path.join(RESULTS_DIR, "cnn_adversarial_summary.csv")
    if os.path.exists(sum_csv):
        df_sum = pd.read_csv(sum_csv)
        df_sum = df_sum.rename(columns={"Mean_Feature_PSNR_dB": "Mean_PSNR_dB","Mean_Feature_MSE": "Mean_MSE"})
        st.dataframe(df_sum[["Attack_Type", "Perturbation_Param", "Accuracy", "Prediction_Change_Pct", "Stego_Evasions", "Clean_False_Alarms", "Mean_PSNR_dB", "Mean_MSE"]])
        
    st.subheader("Transformation Robustness Under Lossy Compression & Filtering")
    trans_csv = os.path.join(RESULTS_DIR, "transformation_results.csv")
    if os.path.exists(trans_csv):
        df_trans = pd.read_csv(trans_csv)
        st.dataframe(df_trans[["Transformation", "Model", "Accuracy", "Sensitivity_Recall", "False_Positive_Rate", "Evasion_Count", "Average_PSNR_dB"]])
        
    trans_img = os.path.join(RESULTS_DIR, "transformation_accuracy_impact.png")
    if os.path.exists(trans_img):
        st.image(trans_img, caption="Detector Accuracy Degradation Across Image Transformations", use_column_width=True)

# ==============================================================================
# SECTION E: PAYLOAD CAPACITY & RECOVERY ANALYSIS
# ==============================================================================
elif selected_section == "E. Payload Capacity & Recovery Analysis":
    st.markdown('<div class="main-title">E. Payload Capacity & Recovery Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Embedding Capacity, Perceptual Distortion (MSE/PSNR), and Bit Error Rate (BER)</div>', unsafe_allow_html=True)
    
    col_demo, col_live = st.columns([1.2, 1])
    with col_demo:
        st.subheader("1. Interactive LSB Embedding & Extraction")
        demo_cover = cv2.imread(os.path.join("data", "clean", "clean_001.png"))
        secret_text = st.text_input("Payload Message:", "DeepDetect University PBL Stego Key 2026")
        msg_bytes = secret_text.encode("utf-8")
        msg_bits = np.unpackbits(np.frombuffer(msg_bytes, dtype=np.uint8))
        
        st.write(f"**Bits**: {len(msg_bits)} bits | **Bytes**: {len(msg_bytes)} bytes")
        
        if st.button("Embed Message into Blue Channel"):
            stego = embed_lsb(demo_cover, msg_bits)
            st.session_state["live_stego"] = stego
            st.session_state["live_bits"] = msg_bits
            st.success("Message embedded into LSB successfully!")
            
        if "live_stego" in st.session_state:
            st.image(cv2.cvtColor(st.session_state["live_stego"], cv2.COLOR_BGR2RGB), caption="Stego Output", width=260)
            if st.button("Extract Embedded Message"):
                ext_bits = extract_lsb(st.session_state["live_stego"], len(st.session_state["live_bits"]))
                errs, ber = calculate_ber(st.session_state["live_bits"], ext_bits)
                decoded_str = np.packbits(ext_bits).tobytes().decode("utf-8", errors="replace")
                st.write(f"**Extracted Text**: `{decoded_str}`")
                st.write(f"**Bit Error Rate (BER)**: `{ber:.6f}` (Incorrect Bits: {errs})")
                
    with col_live:
        st.subheader("2. Quality Degradation vs Payload Rate")
        p_plot = os.path.join(RESULTS_DIR, "payload_vs_psnr.png")
        if os.path.exists(p_plot):
            st.image(p_plot, caption="Stego PSNR vs Payload Capacity (%)", use_column_width=True)
            
    st.subheader("3. Bit Error Rate (BER) Under Attacks")
    b_plot = os.path.join(RESULTS_DIR, "ber_vs_transformations.png")
    if os.path.exists(b_plot):
        st.image(b_plot, caption="BER Under Lossy Transformations (JPEG, Blur, Resizing, Noise)", use_column_width=True)
        
    p_csv = os.path.join(RESULTS_DIR, "payload_robustness.csv")
    if os.path.exists(p_csv):
        df_p = pd.read_csv(p_csv)
        st.dataframe(df_p[["Payload_Rate_Target", "Condition", "Transformed_PSNR_dB", "Incorrect_Bits", "BER", "Message_Recovered_Exactly"]])

# ==============================================================================
# SECTION F: CORE STEGANOGRAPHY METRICS (8 ESSENTIAL VALUES)
# ==============================================================================
elif selected_section == "F. Core Steganography Metrics (8 Essential Values)":
    st.markdown('<div class="main-title">F. Core Image Steganography Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Individual Image Pair Calculation & Visualization of the 8 Essential Pillars</div>', unsafe_allow_html=True)
    
    st.write("""
    This dedicated module computes and presents the **8 essential evaluation metrics** for any image pair:
    1. **BER (Bit Error Rate)**: Ratio of corrupted bits to total embedded bits.
    2. **PSNR (Peak Signal-to-Noise Ratio)**: Perceptual fidelity in decibels (dB).
    3. **MSE (Mean Squared Error)**: Average squared intensity deviation between cover and stego pixels.
    4. **SSIM (Structural Similarity Index)**: Measures structural degradation across luminance, contrast, and structure.
    5. **Payload Capacity**: Size in bits/bytes, Bits Per Pixel (BPP), and channel capacity utilization percentage.
    6. **Robustness**: Recovery survival index under lossy image compression and filtering.
    7. **Imperceptibility**: Transparency rating against standard human visual system (HVS) thresholds.
    8. **Detection Accuracy**: AI-based steganography classification probability and verdict from trained models.
    """)
    
    clean_dir = os.path.join("data", "clean")
    stego_dir = os.path.join("data", "stego")
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    stego_files = sorted([f for f in os.listdir(stego_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    
    st.subheader("1. Select or Upload Image Pair for Evaluation")
    eval_mode = st.radio("Select Input Source:", ["Choose from Dataset Pairs (clean_XXX vs stego_XXX)", "Upload Custom Clean & Stego Images"])
    
    clean_img_bgr = None
    stego_img_bgr = None
    pair_label = ""
    
    if eval_mode == "Choose from Dataset Pairs (clean_XXX vs stego_XXX)":
        pair_idx = st.selectbox("Choose Image Pair:", range(len(clean_files)), format_func=lambda i: f"Pair {i+1:03d}: {clean_files[i]} <-> {stego_files[i]}")
        pair_label = f"Pair {pair_idx+1:03d}: {clean_files[pair_idx]} vs {stego_files[pair_idx]}"
        clean_img_bgr = cv2.imread(os.path.join(clean_dir, clean_files[pair_idx]))
        stego_img_bgr = cv2.imread(os.path.join(stego_dir, stego_files[pair_idx]))
    else:
        col_u1, col_u2 = st.columns(2)
        up_c = col_u1.file_uploader("Upload Original Cover Image", type=["png", "jpg", "jpeg"], key="up_core_c")
        up_s = col_u2.file_uploader("Upload Corresponding Stego Image", type=["png", "jpg", "jpeg"], key="up_core_s")
        if up_c is not None and up_s is not None:
            c_bytes = np.asarray(bytearray(up_c.read()), dtype=np.uint8)
            s_bytes = np.asarray(bytearray(up_s.read()), dtype=np.uint8)
            clean_img_bgr = cv2.imdecode(c_bytes, 1)
            stego_img_bgr = cv2.imdecode(s_bytes, 1)
            pair_label = f"{up_c.name} vs {up_s.name}"
            
    if clean_img_bgr is not None and stego_img_bgr is not None:
        h, w, c = clean_img_bgr.shape
        if stego_img_bgr.shape != clean_img_bgr.shape:
            stego_img_bgr = cv2.resize(stego_img_bgr, (w, h))
            
        clean_rgb = cv2.cvtColor(clean_img_bgr, cv2.COLOR_BGR2RGB)
        stego_rgb = cv2.cvtColor(stego_img_bgr, cv2.COLOR_BGR2RGB)
        
        # Display images side by side
        c_v1, c_v2, c_v3 = st.columns(3)
        c_v1.image(clean_rgb, caption="Original Clean Cover", use_column_width=True)
        c_v2.image(stego_rgb, caption="Stego Image", use_column_width=True)
        
        diff_vis = cv2.absdiff(clean_rgb, stego_rgb)
        diff_amplified = cv2.normalize(diff_vis, None, 0, 255, cv2.NORM_MINMAX)
        c_v3.image(diff_amplified, caption="Amplified Modification Map (xNorm)", use_column_width=True)
        
        # -------------------------------------------------------------
        # COMPUTE THE 8 CORE VALUES
        # -------------------------------------------------------------
        total_pixels = h * w
        max_capacity_bits = total_pixels # 1 bit/pixel blue channel
        
        # 1. MSE
        diff_float = clean_img_bgr.astype(np.float64) - stego_img_bgr.astype(np.float64)
        mse_val = float(np.mean(diff_float ** 2))
        
        # 2. PSNR
        psnr_val = float(10 * np.log10((255.0 ** 2) / mse_val)) if mse_val > 0 else 100.0
        
        # 3. SSIM
        ssim_val, _ = compute_ssim(clean_rgb, stego_rgb, full=True, channel_axis=2)
        ssim_val = float(ssim_val)
        
        # 4. Modified Pixels & Payload Capacity
        # Modified pixels: image difference only
        mod_pixels = int(
            np.count_nonzero(
                np.any(clean_img_bgr != stego_img_bgr, axis=2)
            )
        )

        st.checkbox(
            "This image pair is a verified blue-channel LSB pair",
            value=False,
            key="known_lsb_pair"
        )

        original_message = st.text_input(
            "Enter the original embedded message",
            key="original_payload_text"
        )

        original_bytes = None
        original_bits = None
        if original_message:
            original_bytes = original_message.encode("utf-8")

            original_bits = np.unpackbits(
                np.frombuffer(original_bytes, dtype=np.uint8)
            )

        # Actual payload: only available for a verified LSB pair
        payload_bits = None
        payload_bytes = None
        bpp = None
        capacity_utilization = None

        if st.session_state["known_lsb_pair"] and original_message:
            payload_bits = len(original_bits)
            payload_bytes = len(original_bytes)

            bpp = payload_bits / total_pixels

            capacity_utilization = (
                payload_bits / max_capacity_bits
            ) * 100.0

        st.write(f"Modified pixels: {mod_pixels:,}")
        
        # BER can only be calculated when the original payload is known.
        ber_untouched = None
        ber_jpeg = None
        ber_blur = None
        robustness_score = None

        if st.session_state["known_lsb_pair"] and original_message:

            if len(original_bits) > total_pixels:
                st.error("Payload exceeds the available blue-channel capacity.")
            else:
                extracted_bits = extract_lsb(
                    stego_img_bgr,
                    len(original_bits)
                )

                errors, ber_untouched = calculate_ber(
                    original_bits,
                    extracted_bits
                )

                # JPEG transformation
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                success, encoded = cv2.imencode(
                    ".jpg", stego_img_bgr, encode_param
                )

                if success:
                    jpeg_stego = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

                    jpeg_bits = extract_lsb(
                        jpeg_stego,
                        len(original_bits)
                    )

                    _, ber_jpeg = calculate_ber(
                        original_bits,
                        jpeg_bits
                    )

                # Gaussian blur transformation
                blurred = cv2.GaussianBlur(
                    stego_img_bgr,
                    (3, 3),
                    0
                )

                blur_bits = extract_lsb(
                    blurred,
                    len(original_bits)
                )

                _, ber_blur = calculate_ber(
                    original_bits,
                    blur_bits
                )

                if ber_jpeg is not None and ber_blur is not None:
                    robustness_score = 1.0 - (
                        (ber_jpeg + ber_blur) / 2.0
                    )

                st.success(
                    f"Original payload: {len(original_bits)} bits"
                )

                st.write(f"Untouched BER: {ber_untouched:.6f}")

                if ber_jpeg is not None:
                    st.write(f"JPEG BER: {ber_jpeg:.6f}")

                if ber_blur is not None:
                    st.write(f"Blur BER: {ber_blur:.6f}")

        else:
            st.info(
                "BER and payload recovery are unavailable until you provide "
                "the original message and confirm this is a verified LSB pair."
            )
            
        # 6. Robustness Score (Resilience under transformations: 1.0 - mean BER)
        if ber_jpeg is not None and ber_blur is not None:
            robustness_score = 1.0 - (
                (ber_jpeg + ber_blur) / 2.0
            )
        else:
            robustness_score = None
        
        # 7. Imperceptibility Assessment
        if psnr_val >= 50.0 and ssim_val >= 0.99:
            imperceptibility_label = "Excellent (Indistinguishable from Cover)"
            imperceptibility_badge = "🟢 High Imperceptibility"
        elif psnr_val >= 40.0 and ssim_val >= 0.95:
            imperceptibility_label = "Good (Imperceptible to Human Eye)"
            imperceptibility_badge = "🟡 Standard Imperceptibility"
        else:
            imperceptibility_label = "Moderate / Degraded (Artifacts Present)"
            imperceptibility_badge = "🔴 Low Imperceptibility"
            
        # 8. Detection Accuracy (Stego Probability from Combined LR & CNN)
        gray_stego = cv2.cvtColor(cv2.resize(stego_rgb, (224, 224)), cv2.COLOR_RGB2GRAY)
        feats_stego = extract_all_features(gray_stego)
        
        all_feat_df = pd.read_csv(os.path.join(RESULTS_DIR, "deepdetect_all_features.csv"))
        edge_cols = ["canny_edge_density", "sobel_mean", "sobel_std", "laplacian_mean", "laplacian_variance"]
        res_cols = [c for c in all_feat_df.columns if c.startswith("residual_") or c.startswith("mean_abs") or c.startswith("res_hist_")]
        comb_cols = edge_cols + res_cols
        
        v_comb = np.array([[feats_stego[c] for c in comb_cols]])
        prob_comb = float(models_dict["combined"].predict_proba(v_comb)[0, 1])
        
        prob_cnn = predict_cnn_probability(
            models_dict.get("cnn"),
            stego_img_bgr
        )
            
        ai_detected = (prob_comb >= 0.5)
        
        # -------------------------------------------------------------
        # DISPLAY METRICS DASHBOARD
        # -------------------------------------------------------------
        st.subheader(f"2. Core Values for {pair_label}")
        
        # Row 1: The 4 mathematical distortion & error metrics
        r1_1, r1_2, r1_3, r1_4 = st.columns(4)
        with r1_1:
            if ber_untouched is not None:
                st.metric("1. Bit Error Rate (BER)", f"{ber_untouched:.6f}", delta="0.00% Untouched Extraction")
                jpeg_pct_txt = f"{ber_jpeg*100:.2f}%" if ber_jpeg is not None else "N/A"
                blur_pct_txt = f"{ber_blur*100:.2f}%" if ber_blur is not None else "N/A"
                st.caption(f"Under JPEG-90: `{jpeg_pct_txt}` | Blur: `{blur_pct_txt}`")
            else:
                st.metric("1. Bit Error Rate (BER)", "Unavailable")
                st.caption("Original payload information is required.")
        with r1_2:
            st.metric("2. PSNR (Peak SNR)", f"{psnr_val:.2f} dB", delta="Optimal (>40 dB)" if psnr_val >= 40 else "Degraded")
            st.caption("Fidelity ratio in decibels")
        with r1_3:
            st.metric("3. MSE (Mean Squared Error)", f"{mse_val:.6f}", delta="Minimal Distortion" if mse_val < 1.0 else "Noticeable Drift", delta_color="inverse")
            st.caption("Average squared pixel error")
        with r1_4:
            st.metric("4. SSIM (Structural Similarity)", f"{ssim_val:.4f}", delta="Near Perfect (1.00)" if ssim_val >= 0.98 else "Structural Shift")
            st.caption("Perceptual structural preservation")
            
        # Row 2: Capacity, Robustness, Imperceptibility, Detection Accuracy
        r2_1, r2_2, r2_3, r2_4 = st.columns(4)
        with r2_1:
            if payload_bits is not None:
                st.metric(
                    "5. Payload Capacity",
                    f"{bpp:.6f} bpp",
                    delta=f"{capacity_utilization:.2f}% Capacity Used"
                )
                st.caption(
                    f"{payload_bits:,} bits ({payload_bytes:,} bytes)"
                )
            else:
                st.metric("5. Payload Capacity", "Unavailable")
                st.caption(
                    "Original payload information is required."
                )
        with r2_2:
            if robustness_score is not None:
                st.metric(
                    "6. Robustness Score",
                    f"{robustness_score * 100:.2f}%"
                )
                st.caption("Average recovery score under JPEG and blur tests")
            else:
                st.metric("6. Robustness Score", "Unavailable")
                st.caption("Original payload information is required.")
        with r2_3:
            st.metric("7. Imperceptibility", imperceptibility_badge)
            st.caption(imperceptibility_label)
        with r2_4:
            st.metric("8. Detection Accuracy", f"{prob_comb*100:.1f}% Stego Conf.", 
                      delta="🔴 Stego Detected" if ai_detected else "🟢 Clean / Cover")
            st.caption(
                f"CNN Stego Probability: `{prob_cnn*100:.1f}%`"
                if prob_cnn is not None
                else "CNN model unavailable"
            )
            
        # Summary Table
        st.subheader("3. Detailed Metrics Analysis Table")
        ber_untouched_txt = f"{ber_untouched:.6f}" if ber_untouched is not None else "Unavailable"
        ber_jpeg_txt = f"{ber_jpeg:.6f} ({ber_jpeg*100:.1f}%)" if ber_jpeg is not None else "Unavailable"
        ber_blur_txt = f"{ber_blur:.6f} ({ber_blur*100:.1f}%)" if ber_blur is not None else "Unavailable"
        capacity_txt = f"{capacity_utilization:.2f}%" if capacity_utilization is not None else "an unknown share"
        bpp_txt = f"{bpp:.6f} bpp" if bpp is not None else "Unavailable"
        robustness_txt = f"{robustness_score*100:.2f} / 100.0" if robustness_score is not None else "Unavailable"
        core_summary_df = pd.DataFrame([
            {"Metric": "1. Bit Error Rate (Untouched)", "Calculated Value": ber_untouched_txt, "Standard Target": "0.000000", "Interpretation": "Flawless message recovery on unaltered image"},
            {"Metric": "   Bit Error Rate (JPEG Q=90)", "Calculated Value": ber_jpeg_txt, "Standard Target": "< 0.100000", "Interpretation": "LSB vulnerable to DCT quantization"},
            {"Metric": "   Bit Error Rate (Gaussian Blur)", "Calculated Value": ber_blur_txt, "Standard Target": "< 0.100000", "Interpretation": "Spatial filtering alters least significant bits"},
            {"Metric": "2. Peak Signal-to-Noise Ratio (PSNR)", "Calculated Value": f"{psnr_val:.4f} dB", "Standard Target": "> 40.00 dB", "Interpretation": "High fidelity; imperceptible to human visual system"},
            {"Metric": "3. Mean Squared Error (MSE)", "Calculated Value": f"{mse_val:.6f}", "Standard Target": "< 0.500000", "Interpretation": "Negligible pixel intensity alteration"},
            {"Metric": "4. Structural Similarity Index (SSIM)", "Calculated Value": f"{ssim_val:.6f}", "Standard Target": "> 0.980000", "Interpretation": "Luminance, contrast, and structure preserved"},
            {"Metric": "5. Embedded Payload Bits", "Calculated Value": f"{mod_pixels:,} bits ({mod_pixels//8:,} bytes)", "Standard Target": "Application Dependent", "Interpretation": f"Payload occupies {capacity_txt} of channel capacity"},
            {"Metric": "   Payload Rate (Bits Per Pixel - BPP)", "Calculated Value": bpp_txt, "Standard Target": "0.01 - 0.50 bpp", "Interpretation": "Spatial density of steganographic modification"},
            {"Metric": "6. Robustness Survival Index", "Calculated Value": robustness_txt, "Standard Target": "> 75.0%", "Interpretation": "Composite resistance to lossy channel degradation"},
            {"Metric": "7. Imperceptibility Assessment", "Calculated Value": imperceptibility_label, "Standard Target": "High / Indistinguishable", "Interpretation": f"PSNR={psnr_val:.1f}dB & SSIM={ssim_val:.4f}"},
            {"Metric": "8. Combined LR Stego Probability", "Calculated Value": f"{prob_comb*100:.2f}% (Verdict: {'Stego' if ai_detected else 'Clean'})", "Standard Target": "> 50% for Stego", "Interpretation": "High-frequency spatial noise residual detection"},
            {
                "Metric": "Deep CNN Stego Probability",
                "Calculated Value": (
                    f"{prob_cnn*100:.2f}%"
                    if prob_cnn is not None
                    else "Unavailable"
                ),
                "Standard Target": "> 50% for Stego",
                "Interpretation": (
                    "Deep residual convolutional feature extraction"
                    if prob_cnn is not None
                    else "CNN model could not be loaded"
                )
            }
        ])
        st.dataframe(core_summary_df)

# ==============================================================================
# SECTION G: REPORTS & EXPERIMENT SYNTHESIS
# ==============================================================================
elif selected_section == "G. Reports & Experiment Synthesis":
    st.markdown('<div class="main-title">G. Reports & Experiment Synthesis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Downloadable Artifacts, Result Logs, and Synthesis Summary</div>', unsafe_allow_html=True)
    
    summary_file = os.path.join(RESULTS_DIR, "experiment_summary.txt")
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            st.text_area("PBL Master Experiment Summary", f.read(), height=360)
            
    st.subheader("Download Experimental Datasets & Result CSVs")
    dl_cols = st.columns(4)
    csv_files = [
        ("model_comparison.csv", "Model Comparison"),
        ("cnn_adversarial_summary.csv", "Adversarial Summary"),
        ("transformation_results.csv", "Transformation Results"),
        ("payload_robustness.csv", "Payload Robustness"),
        ("image_metrics.csv", "Image Quality Metrics"),
        ("predictions.csv", "Predictions Log"),
        ("cnn_robustness_results.csv", "Detailed Robustness Log"),
        ("deepdetect_all_features.csv", "Extracted Features")
    ]
    
    for idx, (cf, label) in enumerate(csv_files):
        p = os.path.join(RESULTS_DIR, cf)
        if os.path.exists(p):
            with open(p, "rb") as f:
                dl_cols[idx % 4].download_button(label=f"📥 {label}", data=f, file_name=cf, mime="text/csv")
