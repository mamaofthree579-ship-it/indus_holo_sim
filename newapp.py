import streamlit as st
import urllib.request
import numpy as np
import cv2
import pandas as pd
from skimage.morphology import skeletonize

# ==========================================
# HARMONIC MATRIX: JUST INTONATION FREQUENCIES
# ==========================================
JUST_INTONATION_SCALE = {
    "1:1 (Unison / 0° Base)": 0.0,
    "16:15 (Minor Second)": 90.0 * np.log2(16/15),   # ~ 8.44°
    "9:8 (Major Second)": 90.0 * np.log2(9/8),       # ~ 15.29°
    "6:5 (Minor Third)": 90.0 * np.log2(6/5),         # ~ 24.47°
    "5:4 (Major Third)": 90.0 * np.log2(5/4),         # ~ 28.97°
    "4:3 (Perfect Fourth)": 90.0 * np.log2(4/3),      # ~ 37.37°
    "45:32 (Tritone)": 90.0 * np.log2(45/32),        # ~ 44.18°
    "3:2 (Perfect Fifth)": 90.0 * np.log2(3/2),       # ~ 52.65°
    "8:5 (Minor Sixth)": 90.0 * np.log2(8/5),         # ~ 61.02°
    "5:3 (Major Sixth)": 90.0 * np.log2(5/3),         # ~ 66.37°
    "7:4 (Harmonic Seventh)": 90.0 * np.log2(7/4),    # ~ 72.67°
    "15:8 (Major Seventh)": 90.0 * np.log2(15/8),     # ~ 81.56°
    "2:1 (Octave / 90° Axis)": 90.0                   # 90.0°
}

st.set_page_config(page_title="Indus Script Harmonic Analyzer", layout="wide")
st.title("🏛️ Indus Script Geometric Data Tag Analyzer")
st.markdown("Testing multi-layer 3D data-encoding theories using mathematical fractional harmonic ratios and just intonation grids.")

st.sidebar.header("🎛️ Data Layer Parameters")

# Interactive control sliders
threshold_val = st.sidebar.slider("Simulated Light/Depth Layer (Threshold)", 10, 245, 127)
tolerance_deg = st.sidebar.slider("Harmonic Alignment Tolerance (Degrees)", 0.5, 5.0, 2.0, step=0.5)

image_options = {
    "Fish Sign (Standard Compound)": "https://wikimedia.org",
    "Jar / U-Sign Structure": "https://wikimedia.org"
}
selected_preset = st.sidebar.selectbox("Choose Sample Sign to Analyze", list(image_options.keys()))
image_url = image_options[selected_preset]

@st.cache_data
def download_sign(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        st.error(f"Failed to load image from database: {e}")
        return None

raw_img = download_sign(image_url)

if raw_img is not None:
    # Scale and fit boundaries to a clean normalized 256 square matrix
    raw_img = cv2.resize(raw_img, (256, 256))
    
    # Controlled Threshold slicing: map pixels below value to white foreground (255)
    _, thresh_img = cv2.threshold(raw_img, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    # Force check matrix array formatting
    thresh_img = cv2.convertScaleAbs(thresh_img)

    # Calculate structural bone skeleton profile
    sk_bool = skeletonize(thresh_img > 0)
    sk = (sk_bool.astype(np.uint8)) * 255
    
    # Pull geometric segment paths
    contours, _ = cv2.findContours(sk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_tags = []
    annotated_display = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)

    for c in contours:
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2)
        
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[i-1]
            v2 = pts[i+1] - pts[i]
            
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_product == 0:
                continue
                
            cos_angle = np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))
            angle_quadrant = angle_deg % 90
            
            for interval_name, target_angle in JUST_INTONATION_SCALE.items():
                if abs(angle_quadrant - target_angle) <= tolerance_deg:
                    coord = tuple(pts[i].astype(int))
                    detected_tags.append({
                        "Coordinate": str(coord),
                        "Measured Axis Angle": f"{round(angle_deg, 1)}°",
                        "Harmonic Grid Link": interval_name,
                        "Deviation": f"{round(abs(angle_quadrant - target_angle), 2)}°"
                    })
                    # Highlight localized tags clearly
                    cv2.circle(annotated_display, coord, 4, (255, 0, 0), -1)
                    break

    # Layout using modern width configurations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Processing Layers")
        st.image(thresh_img, caption="Binary Layer Slice (Simulated Lighting Profile)", width="stretch")
        st.image(annotated_display, caption="Tagged Intersections (Blue Dots)", width="stretch")

    with col2:
        st.subheader("🎼 Isolated Geometric Harmonic Matches")
        if detected_tags:
            df_tags = pd.DataFrame(detected_tags).drop_duplicates(subset=["Harmonic Grid Link"])
            st.dataframe(df_tags, width="stretch", hide_index=True)
            st.success(f"Isolated {len(df_tags)} unique physical junctions matching harmonic interval grids.")
        else:
            st.info("No geometry segments match the selected musical ratio tolerances at this threshold layer. Try moving the threshold or tolerance sliders.")
