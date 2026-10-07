"""
Streamlit Web Application: BLE RSSI Indoor Position Prediction & Navigation Dashboard.
Provides interactive localization, floor plan mapping, noise simulation, and model analytics.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

st.set_page_config(
    page_title="BLE Indoor Position Predictor",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics_summary.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
DATA_LABELED_PATH = os.path.join(BASE_DIR, "data", "iBeacon_RSSI_Labeled.csv")
DATA_UNLABELED_PATH = os.path.join(BASE_DIR, "data", "iBeacon_RSSI_Unlabeled.csv")
LAYOUT_IMG_PATH = os.path.join(BASE_DIR, "data", "iBeacon_Layout.jpg")

BEACON_COLS = [f"b30{i:02d}" for i in range(1, 14)]

@st.cache_resource
def load_system_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None
    artifact = joblib.load(MODEL_PATH)
    return artifact

@st.cache_data
def load_data():
    df_lab = pd.read_csv(DATA_LABELED_PATH) if os.path.exists(DATA_LABELED_PATH) else None
    df_unlab = pd.read_csv(DATA_UNLABELED_PATH) if os.path.exists(DATA_UNLABELED_PATH) else None
    df_metrics = pd.read_csv(METRICS_PATH) if os.path.exists(METRICS_PATH) else None
    return df_lab, df_unlab, df_metrics

artifact = load_system_artifacts()
df_labeled, df_unlabeled, df_metrics = load_data()

# ----------------- UI Header -----------------
st.title("📍 BLE Indoor Position Prediction & Localization System")
st.markdown(
    """
    **Project**: Multiclass Indoor Location Prediction from Bluetooth Low Energy (BLE) Signal-Strength Fingerprints  
    **Dataset**: Western Michigan University Waldo Library (13 BLE iBeacons, 105 Spatial Grid Locations)  
    **Algorithms**: KNN, Random Forest, Support Vector Machines, XGBoost, and Stacking/Voting Ensembles.
    """
)
st.divider()

if artifact is None:
    st.warning("⚠️ Model artifact not found yet at `models/best_model.joblib`. Please run the pipeline script (`python run_pipeline.py`) first.")
    st.stop()

model = artifact["model"]
pipeline = artifact["pipeline"]
classes = artifact["classes"]
class_to_coords = artifact["class_to_coords"]
model_name = artifact["model_name"]
metrics_dict = artifact["metrics"]

# ----------------- Sidebar -----------------
st.sidebar.image("https://img.icons8.com/color/96/bluetooth.png", width=64)
st.sidebar.title("Navigation & Controls")
st.sidebar.markdown(f"**Active Model**: `{model_name}`")
st.sidebar.markdown(f"**Test Accuracy**: `{metrics_dict.get('accuracy', 0.0)*100:.2f}%`")
st.sidebar.markdown(f"**Mean Distance Error**: `{metrics_dict.get('mean_distance_error', 0.0):.2f} units`")
st.sidebar.divider()

mode = st.sidebar.radio(
    "Select Operating Mode:",
    [
        "Interactive Fingerprint Prediction",
        "Dataset Sample Explorer & Validation",
        "Noise & Packet Drop Simulator",
        "Model Benchmark & Comparative Analytics"
    ]
)

# Function to render Waldo Library Floor Plan Grid
def render_floor_plan(pred_coord, true_coord=None, pred_loc_name="Predicted", true_loc_name="Actual"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Grid limits based on dataset (Letter D=3 to W=22, Row 1 to 15)
    ax.set_xlim(2, 23)
    ax.set_ylim(0, 16)
    
    # Draw background grid cells
    for x in range(3, 23):
        for y in range(1, 16):
            rect = patches.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, linewidth=0.5, 
                                     edgecolor='#cccccc', facecolor='#f8f9fa', alpha=0.5)
            ax.add_patch(rect)
            
    # Mark True Coordinate if available
    if true_coord is not None:
        tx, ty = true_coord
        ax.scatter(tx, ty, color='#1f77b4', s=250, zorder=5, edgecolors='black', linewidth=1.5, label=f"True: {true_loc_name}")
        ax.annotate(f"True\n({true_loc_name})", (tx, ty), textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=9, fontweight='bold', color='#1f77b4')
        
    # Mark Predicted Coordinate
    px, py = pred_coord
    ax.scatter(px, py, color='#d95f02', marker='*', s=350, zorder=6, edgecolors='black', linewidth=1.5, label=f"Predicted: {pred_loc_name}")
    ax.annotate(f"Pred\n({pred_loc_name})", (px, py), textcoords="offset points", xytext=(0, -22),
                ha='center', fontsize=9, fontweight='bold', color='#d95f02')
    
    # Draw error line if true coordinate exists
    if true_coord is not None:
        ax.plot([tx, px], [ty, py], color='red', linestyle='--', linewidth=2.0, zorder=4, label='Error Vector')
        dist = np.sqrt((px - tx)**2 + (py - ty)**2)
        mid_x, mid_y = (px + tx) / 2.0, (py + ty) / 2.0
        ax.annotate(f"Error: {dist:.2f} units", (mid_x, mid_y), textcoords="offset points", xytext=(10, 5),
                    fontsize=9, fontweight='bold', color='red', bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6))
        
    # Formatting
    col_labels = [chr(ord('A') + x) for x in range(3, 23)]
    ax.set_xticks(range(3, 23))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(1, 16))
    ax.set_yticklabels(range(1, 16), fontsize=9)
    
    ax.set_xlabel("Library Grid Column (East-West)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Library Grid Row (North-South)", fontsize=11, fontweight='bold')
    ax.set_title("Waldo Library Indoor Localization Grid Map", fontsize=13, fontweight='bold')
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    return fig

# ----------------- MODE 1: Interactive Fingerprint Prediction -----------------
if mode == "Interactive Fingerprint Prediction":
    st.subheader("🎛️ Real-Time Signal Fingerprint Simulation")
    st.markdown("Adjust individual iBeacon RSSI readings (-200 dBm indicates Out-of-Range; -50 dBm indicates immediate proximity):")
    
    col_preset, col_clear = st.columns([3, 1])
    preset_choice = col_preset.selectbox(
        "Load Typical Room Preset:",
        ["Custom Manual Input", "Preset 1: Near Beacon 3002 (e.g. Location K04)", "Preset 2: Central Hallway (Location J06)", "Preset 3: Peripheral East (Location T15)"]
    )
    
    default_vals = {f"b30{i:02d}": -200 for i in range(1, 14)}
    if preset_choice == "Preset 1: Near Beacon 3002 (e.g. Location K04)":
        default_vals["b3002"] = -62
        default_vals["b3004"] = -84
    elif preset_choice == "Preset 2: Central Hallway (Location J06)":
        default_vals["b3002"] = -74
        default_vals["b3006"] = -79
        default_vals["b3004"] = -81
    elif preset_choice == "Preset 3: Peripheral East (Location T15)":
        default_vals["b3013"] = -72
        default_vals["b3008"] = -88

    # Create 3 columns for 13 beacon sliders
    cols = st.columns(3)
    user_rssi = {}
    for i in range(1, 14):
        b_name = f"b30{i:02d}"
        col_idx = (i - 1) % 3
        with cols[col_idx]:
            user_rssi[b_name] = st.slider(
                f"Beacon {b_name} RSSI (dBm):",
                min_value=-200,
                max_value=-40,
                value=int(default_vals[b_name]),
                step=1
            )
            
    # Form inference dataframe
    input_df = pd.DataFrame([user_rssi])
    X_input, _ = pipeline.transform(input_df)
    
    # Predict
    pred_idx = model.predict(X_input)[0]
    pred_loc = pipeline.inverse_transform_labels(np.array([pred_idx]))[0]
    pred_coord = pipeline.class_to_coords[pred_idx]
    
    # Probabilities
    proba = model.predict_proba(X_input)[0] if hasattr(model, "predict_proba") else None
    
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.success(f"### Predicted Location: **{pred_loc}**")
        st.markdown(f"- **Grid Coordinates**: Column `X={pred_coord[0]}` ({chr(ord('A') + pred_coord[0])}), Row `Y={pred_coord[1]}`")
        if proba is not None:
            conf = float(np.max(proba)) * 100.0
            st.metric("Model Confidence", f"{conf:.1f}%")
            
            # Top 3 candidates
            top3_idx = np.argsort(proba)[-3:][::-1]
            st.markdown("**Top-3 Candidate Cells:**")
            for rank, c_idx in enumerate(top3_idx):
                c_loc = classes[c_idx]
                c_p = proba[c_idx] * 100.0
                st.write(f"{rank+1}. **{c_loc}**: {c_p:.1f}%")
                
    with res_col2:
        st.pyplot(render_floor_plan(pred_coord, pred_loc_name=pred_loc))

# ----------------- MODE 2: Dataset Sample Explorer -----------------
elif mode == "Dataset Sample Explorer & Validation":
    st.subheader("🔍 Explore Historical Labeled Test Fingerprints")
    st.markdown("Select a ground-truth measurement from the test partition to inspect prediction accuracy and physical localization error.")
    
    if df_labeled is not None:
        sample_idx = st.slider("Select Sample Index:", 0, len(df_labeled) - 1, 42)
        sample_row = df_labeled.iloc[[sample_idx]].copy()
        
        true_loc = sample_row['location'].values[0]
        true_coord = pipeline.loc_to_coords.get(true_loc, (0, 0))
        
        X_sample, _ = pipeline.transform(sample_row)
        pred_idx = model.predict(X_sample)[0]
        pred_loc = pipeline.inverse_transform_labels(np.array([pred_idx]))[0]
        pred_coord = pipeline.class_to_coords[pred_idx]
        
        dist_err = np.sqrt((pred_coord[0] - true_coord[0])**2 + (pred_coord[1] - true_coord[1])**2)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("True Ground-Truth Location", true_loc)
        col2.metric("Predicted Location", pred_loc)
        col3.metric("Localization Distance Error", f"{dist_err:.2f} units", delta=f"{'Exact Match' if dist_err == 0 else 'Offset'}")
        
        st.pyplot(render_floor_plan(pred_coord, true_coord, pred_loc_name=pred_loc, true_loc_name=true_loc))
        
        st.markdown("**Active Detected Beacons in this Sample:**")
        active_beacons = {b: sample_row[b].values[0] for b in BEACON_COLS if sample_row[b].values[0] > -200}
        st.json(active_beacons)

# ----------------- MODE 3: Noise & Packet Drop Simulator -----------------
elif mode == "Noise & Packet Drop Simulator":
    st.subheader("📶 Wireless Channel Impairment & Noise Robustness Simulator")
    st.markdown(
        """
        Simulate multipath fading, interference, and packet collisions by injecting Gaussian RSSI jitter 
        and random beacon dropouts.
        """
    )
    
    noise_sigma = st.slider("Gaussian Multipath Noise Std Dev (dBm):", 0.0, 20.0, 5.0, 1.0)
    drop_pct = st.slider("Random Packet Drop Rate (%):", 0, 60, 10, 5) / 100.0
    
    if st.button("Evaluate Robustness on Test Batch (100 samples)"):
        test_sample = df_labeled.sample(n=min(100, len(df_labeled)), random_state=42).copy()
        
        # Inject noise and drop
        from src.noise_experiment import inject_gaussian_noise, inject_signal_drop
        noisy_df = inject_gaussian_noise(test_sample, noise_std=noise_sigma)
        noisy_df = inject_signal_drop(noisy_df, drop_rate=drop_pct)
        
        X_eval, y_true = pipeline.transform(noisy_df)
        preds = model.predict(X_eval)
        
        acc = np.mean(preds == y_true)
        true_coords = np.array([pipeline.class_to_coords[i] for i in y_true])
        pred_coords = np.array([pipeline.class_to_coords[i] for i in preds])
        mean_err = np.mean(np.sqrt(np.sum((pred_coords - true_coords)**2, axis=1)))
        
        c1, c2 = st.columns(2)
        c1.metric("Degraded Accuracy", f"{acc*100:.1f}%")
        c2.metric("Mean Localization Error", f"{mean_err:.2f} units")
        
    if os.path.exists(f"{FIGURES_DIR}/noise_robustness.png"):
        st.image(f"{FIGURES_DIR}/noise_robustness.png", caption="Systematic Noise Degradation Analysis across Models")

# ----------------- MODE 4: Model Benchmark & Comparative Analytics -----------------
elif mode == "Model Benchmark & Comparative Analytics":
    st.subheader("📊 Comparative Evaluation of 6 Machine Learning Algorithms")
    
    if df_metrics is not None:
        st.dataframe(df_metrics.style.highlight_min(subset=['mean_distance_error'], color='#c7e9c0')
                                   .highlight_max(subset=['accuracy'], color='#c7e9c0'), use_container_width=True)
        
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(f"{FIGURES_DIR}/error_cdf.png"):
            st.image(f"{FIGURES_DIR}/error_cdf.png", caption="Cumulative Distribution Function (CDF) of Localization Error")
    with c2:
        if os.path.exists(f"{FIGURES_DIR}/feature_importance.png"):
            st.image(f"{FIGURES_DIR}/feature_importance.png", caption="Feature Influence Analysis (Mutual Information & RF Importance)")
            
    c3, c4 = st.columns(2)
    with c3:
        if os.path.exists(f"{FIGURES_DIR}/permutation_importance.png"):
            st.image(f"{FIGURES_DIR}/permutation_importance.png", caption="Permutation Feature Importance on Validation Set")
    with c4:
        if os.path.exists(f"{FIGURES_DIR}/error_spatial_map.png"):
            st.image(f"{FIGURES_DIR}/error_spatial_map.png", caption="Spatial Heatmap of Floor Grid Localization Errors")
