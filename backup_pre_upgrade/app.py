import os
import sys

# Ensure local matplotlib configdir
os.environ["MPLCONFIGDIR"] = os.path.abspath(os.path.join(".", ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ["KERAS_BACKEND"] = "jax"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import joblib

# DeepDetect modules
from residual_analysis import (
    compute_residual, extract_edge_features, extract_residual_features,
    extract_all_features, MODELS_DIR, RESULTS_DIR
)
from payload_robustness import calculate_mse, calculate_psnr, calculate_ber, embed_lsb, extract_lsb

st.set_page_config(
    page_title="DeepDetect — AI Steganalysis & Robustness",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2563EB; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🛡️ DeepDetect System")
st.sidebar.markdown("**AI Steganalysis, Adversarial Attacks & Robustness**")

sections = [
    "1. Project Overview",
    "2. Upload & Live Steganalysis",
    "3. LSB Embedding & Extraction",
    "4. Image Quality Analysis (MSE & PSNR)",
    "5. Capacity & Payload Analysis",
    "6. BER & Message Recovery",
    "7. Edge-Based Steganalysis",
    "8. Residual-Based Steganalysis",
    "9. Logistic Regression Model Comparison",
    "10. CNN Deep Steganalysis",
    "11. Image Transformation Experiments",
    "12. CNN Adversarial Robustness (FGSM)",
    "13. Final Experiment Summary & Reports"
]

selected_section = st.sidebar.radio("Navigation / PBL Modules", sections)

# Helper function to load models
_CACHED_MODELS = None

def load_models():
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

models_dict = load_models()

# -----------------------------
# 1. PROJECT OVERVIEW
# -----------------------------
if selected_section == "1. Project Overview":
    st.markdown('<div class="main-title">DeepDetect: AI-Based Image Steganalysis & Robustness System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">University CSE Capstone PBL Project</div>', unsafe_allow_html=True)
    
    st.write("""
    **DeepDetect** is an end-to-end framework designed to inspect, detect, and analyze high-frequency steganographic 
    modifications (such as Least Significant Bit replacement) within digital images. The system investigates whether 
    residual spatial noise filtering and deep convolutional features outperform classical spatial edge representations, 
    evaluating detection performance under severe lossy image transformations and white-box adversarial attacks (FGSM).
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**Dataset Size**\n\n100 Clean & 100 Stego paired scenes (200 total images)")
    with col2:
        st.success("**Models Built**\n\n3 Logistic Regression models + 1 Deep CNN Classifier")
    with col3:
        st.warning("**Attacks Evaluated**\n\nJPEG, Blur, Resizing, Noise, Cropping & FGSM")
    with col4:
        st.error("**Data Leakage Prevention**\n\nZero leakage across paired scenes via GroupShuffleSplit")

    st.subheader("System Architecture")
    st.markdown("""
    - **Cover / Stego Pipeline**: Blue-channel LSB embedding with deterministic bit allocations and ground-truth validation.
    - **Feature Extraction**: Dual feature bank comprising spatial edge gradients (Sobel, Laplacian, Canny edge density) 
      and high-frequency noise residuals ($R = I - \text{GaussianBlur}(I)$).
    - **Detectors**: Edge-Only LR, Residual-Only LR, Combined Edge+Residual LR, and a deep 3-block Conv2D network.
    - **Adversarial Robustness**: Fast Gradient Sign Method (FGSM) evaluating perturbation strength ($\epsilon$) tradeoffs.
    """)

# -----------------------------
# 2. UPLOAD & LIVE STEGANALYSIS
# -----------------------------
elif selected_section == "2. Upload & Live Steganalysis":
    st.header("Upload Original / Stego Image for Live Inference")
    st.write("Upload an image to extract edge/residual features and run predictions across all trained detectors.")
    
    uploaded_file = st.file_uploader("Choose an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    col_demo, _ = st.columns([1, 2])
    use_sample = col_demo.checkbox("Or use existing dataset sample image", value=(uploaded_file is None))
    
    image = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
    elif use_sample:
        sample_path = os.path.join("data", "stego", "stego_001.png")
        if os.path.exists(sample_path):
            image = cv2.imread(sample_path)
            st.caption(f"Loaded demo image: `{sample_path}`")
            
    if image is not None:
        h, w, _ = image.shape
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb_img, (224, 224))
        gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
        
        col_img, col_pred = st.columns([1, 1.2])
        with col_img:
            st.image(rgb_img, caption=f"Analyzed Image ({w}x{h})", use_column_width=True)
            
        with col_pred:
            st.subheader("Model Predictions & Confidences")
            feats = extract_all_features(gray)
            
            # Predict LR
            feat_df = pd.read_csv(os.path.join(RESULTS_DIR, "deepdetect_all_features.csv"))
            edge_cols = ["canny_edge_density", "sobel_mean", "sobel_std", "laplacian_mean", "laplacian_variance"]
            res_cols = [c for c in feat_df.columns if c.startswith("residual_") or c.startswith("mean_abs") or c.startswith("res_hist_")]
            comb_cols = edge_cols + res_cols
            
            v_edge = np.array([[feats[c] for c in edge_cols]])
            v_res = np.array([[feats[c] for c in res_cols]])
            v_comb = np.array([[feats[c] for c in comb_cols]])
            
            prob_edge = models_dict["edge"].predict_proba(v_edge)[0, 1]
            prob_res = models_dict["residual"].predict_proba(v_res)[0, 1]
            prob_comb = models_dict["combined"].predict_proba(v_comb)[0, 1]
            
            pred_cnn_prob = 0.5
            if models_dict["cnn"] is not None:
                cnn_in = np.expand_dims(cv2.resize(rgb_img, (128, 128)).astype(np.float32) / 255.0, axis=0)
                pred_cnn_prob = float(models_dict["cnn"].predict(cnn_in, verbose=0)[0, 0])
                
            pred_df = pd.DataFrame({
                "Model": ["Edge-Only LR", "Residual-Only LR", "Combined Edge+Residual LR", "Deep CNN Detector"],
                "Confidence (Stego %)": [f"{prob_edge*100:.1f}%", f"{prob_res*100:.1f}%", f"{prob_comb*100:.1f}%", f"{pred_cnn_prob*100:.1f}%"],
                "Decision": [
                    "🔴 Stego" if prob_edge >= 0.5 else "🟢 Clean",
                    "🔴 Stego" if prob_res >= 0.5 else "🟢 Clean",
                    "🔴 Stego" if prob_comb >= 0.5 else "🟢 Clean",
                    "🔴 Stego" if pred_cnn_prob >= 0.5 else "🟢 Clean"
                ]
            })
            st.dataframe(pred_df)
            
            st.metric("Combined Detector Probability", f"{prob_comb*100:.2f}%", 
                      delta="Steganography Detected" if prob_comb >= 0.5 else "Cover Image (Clean)")

# -----------------------------
# 3. LSB EMBEDDING & EXTRACTION
# -----------------------------
elif selected_section == "3. LSB Embedding & Extraction":
    st.header("Interactive LSB Embedding & Extraction")
    st.write("Embed custom text payload into blue channel LSB and verify extraction integrity.")
    
    demo_cover_path = os.path.join("data", "clean", "clean_001.png")
    cover = cv2.imread(demo_cover_path)
    
    col_embed, col_extract = st.columns(2)
    with col_embed:
        user_msg = st.text_input("Enter Secret Message:", "DeepDetect University PBL Stego Key 2026")
        msg_bytes = user_msg.encode("utf-8")
        msg_bits = np.unpackbits(np.frombuffer(msg_bytes, dtype=np.uint8))
        
        st.write(f"**Payload Bits**: {len(msg_bits)} bits ({len(msg_bytes)} bytes)")
        
        if st.button("Embed Message"):
            stego = embed_lsb(cover, msg_bits)
            st.session_state["live_stego"] = stego
            st.session_state["live_bits"] = msg_bits
            st.success("Message embedded into blue channel LSB successfully!")
            
    with col_extract:
        if "live_stego" in st.session_state:
            st.image(cv2.cvtColor(st.session_state["live_stego"], cv2.COLOR_BGR2RGB), caption="Generated Stego Image", width=250)
            if st.button("Extract Embedded Secret Message"):
                extracted_bits = extract_lsb(st.session_state["live_stego"], len(st.session_state["live_bits"]))
                errs, ber = calculate_ber(st.session_state["live_bits"], extracted_bits)
                byte_arr = np.packbits(extracted_bits).tobytes()
                try:
                    decoded_msg = byte_arr.decode("utf-8")
                except Exception:
                    decoded_msg = str(byte_arr)
                    
                st.write(f"**Extracted Text**: `{decoded_msg}`")
                st.write(f"**Bit Error Rate (BER)**: `{ber:.6f}` (Incorrect Bits: {errs})")
        else:
            st.info("Click 'Embed Message' first.")

# -----------------------------
# 4. IMAGE QUALITY ANALYSIS
# -----------------------------
elif selected_section == "4. Image Quality Analysis (MSE & PSNR)":
    st.header("Image Quality Metrics: MSE and PSNR")
    st.write("Evaluating distortion between pristine clean covers and corresponding stego images.")
    
    csv_path = os.path.join(RESULTS_DIR, "image_metrics.csv")
    if os.path.exists(csv_path):
        df_metrics = pd.read_csv(csv_path)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Stego MSE", f"{df_metrics['MSE'].mean():.4f}")
        c2.metric("Mean Stego PSNR", f"{df_metrics['PSNR_dB'].mean():.2f} dB")
        c3.metric("Evaluated Image Pairs", f"{len(df_metrics)}")
        
        st.subheader("Dataset Image Quality Table")
        st.dataframe(df_metrics.head(20))
    else:
        st.warning("image_metrics.csv not found. Please run generate_reports.py.")

# -----------------------------
# 5. CAPACITY & PAYLOAD ANALYSIS
# -----------------------------
elif selected_section == "5. Capacity & Payload Analysis":
    st.header("Payload Capacity Analysis")
    st.write("Relationship between embedding capacity utilization, bits per pixel (BPP), MSE, and PSNR.")
    
    csv_path = os.path.join(RESULTS_DIR, "payload_robustness.csv")
    if os.path.exists(csv_path):
        df_p = pd.read_csv(csv_path)
        untouched = df_p[df_p["Condition"] == "Untouched"]
        
        st.dataframe(untouched[["Payload_Rate_Target", "Payload_Bits", "Payload_Bytes", "Bits_Per_Pixel", "Stego_MSE", "Stego_PSNR_dB"]])
        
        img_path = os.path.join(RESULTS_DIR, "payload_vs_psnr.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Stego PSNR Degradation vs. Payload Rate (%)", use_column_width=True)
    else:
        st.warning("payload_robustness.csv not found.")

# -----------------------------
# 6. BER & MESSAGE RECOVERY
# -----------------------------
elif selected_section == "6. BER & Message Recovery":
    st.header("Bit Error Rate (BER) & Message Recovery Under Attacks")
    st.markdown("""
    Formula:  
    $$\\text{BER} = \\frac{\\text{Incorrect Recovered Bits}}{\\text{Total Embedded Bits}}$$
    """)
    
    img_path = os.path.join(RESULTS_DIR, "ber_vs_transformations.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Bit Error Rate (BER) Under JPEG, Blur, Resize, and Noise", use_column_width=True)
        
    csv_path = os.path.join(RESULTS_DIR, "payload_robustness.csv")
    if os.path.exists(csv_path):
        df_p = pd.read_csv(csv_path)
        st.dataframe(df_p[["Payload_Rate_Target", "Condition", "Transformed_PSNR_dB", "Incorrect_Bits", "BER", "Message_Recovered_Exactly"]])

# -----------------------------
# 7. EDGE-BASED STEGANALYSIS
# -----------------------------
elif selected_section == "7. Edge-Based Steganalysis":
    st.header("Edge Feature Representations")
    st.write("Classic spatial edge extractors: Sobel gradient magnitude, Laplacian variance, and Canny edge density.")
    
    sample_edge_img = os.path.join("outputs", "edge_analysis_result.png")
    if os.path.exists(sample_edge_img):
        st.image(sample_edge_img, caption="Edge Decomposition (Sobel, Laplacian, Canny)", use_column_width=True)

# -----------------------------
# 8. RESIDUAL-BASED STEGANALYSIS
# -----------------------------
elif selected_section == "8. Residual-Based Steganalysis":
    st.header("Residual-Based Feature Extraction: $R = I - \\text{GaussianBlur}(I)$")
    st.markdown("""
    Spatial noise residuals isolate high-frequency components from low-frequency structural content.
    Because LSB steganography acts as additive pseudorandom noise, the residual image $R$ exhibits distinctive
    statistical anomalies in its variance, kurtosis, and histogram distribution.
    """)
    
    res_vis = os.path.join(RESULTS_DIR, "residual_visualization.png")
    if os.path.exists(res_vis):
        st.image(res_vis, caption="Clean vs. Stego Noise Residuals", use_column_width=True)

# -----------------------------
# 9. LOGISTIC REGRESSION COMPARISON
# -----------------------------
elif selected_section == "9. Logistic Regression Model Comparison":
    st.header("Steganalysis Model Evaluation & Comparison")
    st.write("Edge-Only vs. Residual-Only vs. Combined Edge+Residual Logistic Regression (Held-out Group Test Split).")
    
    comp_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        st.dataframe(df_comp[["Model", "Accuracy", "Precision", "Sensitivity_Recall", "Specificity", "F1_Score", "ROC_AUC", "FPR"]])
        
        c1, c2 = st.columns(2)
        with c1:
            bar_img = os.path.join(RESULTS_DIR, "model_comparison_barchart.png")
            if os.path.exists(bar_img):
                st.image(bar_img, caption="Model Performance Metrics Comparison", use_column_width=True)
        with c2:
            roc_img = os.path.join(RESULTS_DIR, "roc_curve_comparison.png")
            if os.path.exists(roc_img):
                st.image(roc_img, caption="ROC Curves Comparison (Edge vs Residual vs Combined)", use_column_width=True)
                
        st.subheader("Confusion Matrices")
        cm_cols = st.columns(3)
        cm_files = [
            ("confusion_matrix_Edge_Only_LogisticRegression.png", "Edge-Only LR"),
            ("confusion_matrix_Residual_Only_LogisticRegression.png", "Residual-Only LR"),
            ("confusion_matrix_Combined_LogisticRegression.png", "Combined LR")
        ]
        for col, (f_name, title) in zip(cm_cols, cm_files):
            p = os.path.join(RESULTS_DIR, f_name)
            if os.path.exists(p):
                col.image(p, caption=title, use_column_width=True)

# -----------------------------
# 10. CNN DEEP STEGANALYSIS
# -----------------------------
elif selected_section == "10. CNN Deep Steganalysis":
    st.header("Deep CNN Architecture for Image Steganalysis")
    st.markdown("""
    - **Architecture**: 3-stage Conv2D + BatchNormalization + MaxPooling2D feature extractor + Dense classification head.
    - **Loss & Optimizer**: Binary Cross-Entropy with Adam Optimizer ($lr=0.001$).
    - **Model File**: `models/cnn_model.keras`
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        curves_p = os.path.join(RESULTS_DIR, "cnn_training_curves.png")
        if os.path.exists(curves_p):
            st.image(curves_p, caption="CNN Training & Validation Accuracy/Loss Curves", use_column_width=True)
    with c2:
        cm_p = os.path.join(RESULTS_DIR, "confusion_matrix_cnn.png")
        if os.path.exists(cm_p):
            st.image(cm_p, caption="CNN Test Confusion Matrix", use_column_width=True)

# -----------------------------
# 11. IMAGE TRANSFORMATION EXPERIMENTS
# -----------------------------
elif selected_section == "11. Image Transformation Experiments":
    st.header("Image Transformation Robustness & AI Evasion")
    st.write("Assessing detector performance degradation and evasion under JPEG compression, blur, resizing, noise, and cropping.")
    
    t_csv = os.path.join(RESULTS_DIR, "transformation_results.csv")
    if os.path.exists(t_csv):
        df_t = pd.read_csv(t_csv)
        st.dataframe(df_t[["Transformation", "Model", "Accuracy", "Sensitivity_Recall", "False_Positive_Rate", "Evasion_Count", "Average_PSNR_dB"]])
        
        t_plot = os.path.join(RESULTS_DIR, "transformation_accuracy_impact.png")
        if os.path.exists(t_plot):
            st.image(t_plot, caption="Detector Accuracy Degradation Across Image Transformations", use_column_width=True)

# -----------------------------
# 12. CNN ADVERSARIAL ROBUSTNESS
# -----------------------------
elif selected_section == "12. CNN Adversarial Robustness (FGSM)":
    st.header("CNN Adversarial Robustness (Fast Gradient Sign Method)")
    st.markdown("""
    **FGSM Formula**:
    $$x_{\\text{adv}} = \\text{clip}\\left(x + \\epsilon \\cdot \\text{sign}\\left(\\nabla_x \\mathcal{L}(x, y)\\right), 0, 1\\right)$$
    - **$\\epsilon$ (Epsilon)**: Perturbation budget controlling the maximum per-pixel shift.
    - **Objective**: Induce model misclassification while preserving human perceptual quality ($>30$ dB PSNR).
    """)
    
    adv_csv = os.path.join(RESULTS_DIR, "cnn_adversarial_summary.csv")
    if os.path.exists(adv_csv):
        df_adv = pd.read_csv(adv_csv)
        st.dataframe(df_adv)
        
        adv_plot = os.path.join(RESULTS_DIR, "cnn_adversarial_tradeoff.png")
        if os.path.exists(adv_plot):
            st.image(adv_plot, caption="Adversarial Tradeoff: Prediction Shift vs Image Quality (PSNR)", use_column_width=True)

# -----------------------------
# 13. FINAL SUMMARY & REPORTS
# -----------------------------
elif selected_section == "13. Final Experiment Summary & Reports":
    st.header("Final Experiment Summary & Downloadable Reports")
    
    summary_file = os.path.join(RESULTS_DIR, "experiment_summary.txt")
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_text = f.read()
        st.text_area("PBL Experiment Summary", summary_text, height=350)
        
    st.subheader("Download Generated Result CSVs")
    for csv_name in [
        "model_comparison.csv", "transformation_results.csv", "payload_robustness.csv",
        "cnn_adversarial_summary.csv", "cnn_robustness_results.csv", "image_metrics.csv", "predictions.csv"
    ]:
        p = os.path.join(RESULTS_DIR, csv_name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                st.download_button(label=f"📥 Download {csv_name}", data=f, file_name=csv_name, mime="text/csv")
