import streamlit as st
import numpy as np
import cv2
import pandas as pd
from skimage.morphology import skeletonize
from streamlit_drawable_canvas import st_canvas  # Requires: pip install streamlit-drawable-canvas

# ==========================================
# HARMONIC MATRIX: JUST INTONATION MATRICES
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
st.markdown("Testing multi-layer 3D data-encoding theories using mathematical fractional harmonic ratios.")

# --- Side Controls ---
st.sidebar.header("🎛️ Analysis Controls")
input_method = st.sidebar.radio("Data Input Choice", ["Draw Symbol Freehand", "Upload Image File"])
tolerance_deg = st.sidebar.slider("Harmonic Alignment Tolerance (Degrees)", 0.5, 5.0, 2.0, step=0.5)

thresh_img = None

# --- Method A: Draw Interactive Canvas ---
if input_method == "Draw Symbol Freehand":
    st.subheader("✏️ Draw an Indus Sign Layer Variant below:")
    st.caption("Draw structural segments with your mouse. Use sharp adjustments to test if junctions calculate musical ratios.")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=6,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=256,
        width=256,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if canvas_result.image_data is not None:
        # Convert the drawing directly into a binary layer matrix
        rgba_img = canvas_result.image_data
        thresh_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2GRAY)

# --- Method B: Local File Uploader ---
else:
    st.subheader("📁 Upload a Local Script Extraction:")
    uploaded_file = st.file_uploader("Select PNG or JPG image asset", type=["png", "jpg", "jpeg"])
    threshold_val = st.sidebar.slider("Simulated Light/Depth Layer (Threshold)", 10, 245, 127)
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        # Flatten any transparency channel blocks safely onto a white background canvas
        if img is not None and img.shape[-1] == 4:
            alpha = img[:, :, 3] / 255.0
            bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
            for c in range(3):
                bg[:, :, c] = (alpha * img[:, :, c] + (1.0 - alpha) * 255).astype(np.uint8)
            gray_img = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        elif img is not None and len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
            
        gray_img = cv2.resize(gray_img, (256, 256))
        _, thresh_img = cv2.threshold(gray_img, threshold_val, 255, cv2.THRESH_BINARY_INV)

# --- Process Extracted Target Grids ---
if thresh_img is not None and np.sum(thresh_img) > 0:
    thresh_img = cv2.convertScaleAbs(thresh_img)
    
    # Structural bone path processing
    sk_bool = skeletonize(thresh_img > 0)
    sk = (sk_bool.astype(np.uint8)) * 255
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
                        "Coordinate Location": str(coord),
                        "Measured Axis Angle": f"{round(angle_deg, 1)}°",
                        "Harmonic Grid Link": interval_name,
                        "Deviation": f"{round(abs(angle_quadrant - target_angle), 2)}°"
                    })
                    cv2.circle(annotated_display, coord, 4, (0, 255, 0), -1) # Green anchor tagging dots
                    break

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Structural Processing")
        st.image(annotated_display, caption="Detected Geometric Data Tags (Green Dots)", use_container_width=True)
    with col2:
        st.subheader("🎼 Tuning Matrix Alignments")
        if detected_tags:
            df_tags = pd.DataFrame(detected_tags).drop_duplicates(subset=["Harmonic Grid Link"])
            st.dataframe(df_tags, use_container_width=True, hide_index=True)
            st.success(f"Isolated {len(df_tags)} intersections confirming to geometric data scales.")
        else:
            st.info("No junctions fit the tuning scale ratios at this tolerance level. Adjust vectors or drawing lines.")
else:
    st.warning("Waiting for symbol layout inputs. Draw or upload a script component above.")
