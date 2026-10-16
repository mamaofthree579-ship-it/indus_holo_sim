import streamlit as st
import urllib.request
import numpy as np
import cv2
import pandas as pd
from skimage.morphology import skeletonize

# Print directly to your terminal command-line window to see where it stops
print("\n>>> DEBUG: Streamlit script execution kicked off successfully.")

st.set_page_config(page_title="Indus Script Analyzer", layout="wide")
st.title("🏛️ Indus Script Geometric Data Tag Analyzer")
st.text("If you can see this text, the Streamlit frontend layout is rendering correctly.")

# ==========================================
# HARMONIC MATRIX REFERENCE
# ==========================================
JUST_INTONATION_SCALE = {
    "1:1 (0° Base)": 0.0,
    "4:3 (Perfect Fourth)": 90.0 * np.log2(4/3),      # ~ 37.37°
    "3:2 (Perfect Fifth)": 90.0 * np.log2(3/2),       # ~ 52.65°
    "2:1 (Octave / 90° Axis)": 90.0
}

st.sidebar.header("🎛️ Parameters")
threshold_val = st.sidebar.slider("Simulated Light Layer", 10, 245, 200)
tolerance_deg = st.sidebar.slider("Harmonic Tolerance (Degrees)", 0.5, 5.0, 2.0)

image_url = "https://wikimedia.org"

@st.cache_data
def download_and_flatten_sign(url):
    print(f">>> DEBUG: Attempting to download network resource: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                print(">>> DEBUG ERROR: OpenCV failed to decode downloaded image bytes.")
                return None
                
            # If image has an Alpha transparent channel, flatten it manually
            if img.shape[-1] == 4:
                alpha = img[:, :, 3] / 255.0
                bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
                for c in range(3):
                    bg[:, :, c] = (alpha * img[:, :, c] + (1.0 - alpha) * 255).astype(np.uint8)
                return cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    except Exception as e:
        print(f">>> DEBUG CRASH in download function: {e}")
        return None

gray_img = download_and_flatten_sign(image_url)

if gray_img is not None:
    print(f">>> DEBUG: Image loaded cleanly. Array shape matrix dimensions: {gray_img.shape}")
    gray_img = cv2.resize(gray_img, (256, 256))
    
    # Process threshold slice
    _, thresh_img = cv2.threshold(gray_img, threshold_val, 255, cv2.THRESH_BINARY_INV)
    thresh_img = cv2.convertScaleAbs(thresh_img)

    # Skeletonize layout strokes
    sk_bool = skeletonize(thresh_img > 0)
    sk = (sk_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(sk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f">>> DEBUG: Isolated {len(contours)} trace vector path loops from skeleton.")
    
    detected_tags = []
    annotated_display = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)

    for c in contours:
        pts = c.reshape(-1, 2)
        if len(pts) < 3:
            continue
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[i-1]
            v2 = pts[i+1] - pts[i]
            norm_p = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_p == 0:
                continue
            cos_a = np.clip(np.dot(v1, v2) / norm_p, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_a))
            angle_quadrant = angle_deg % 90
            
            for name, target in JUST_INTONATION_SCALE.items():
                if abs(angle_quadrant - target) <= tolerance_deg:
                    coord = tuple(pts[i].astype(int))
                    detected_tags.append({
                        "Coordinate": str(coord),
                        "Measured Angle": f"{round(angle_deg, 1)}°",
                        "Harmonic Grid Link": name,
                    })
                    cv2.circle(annotated_display, coord, 4, (255, 0, 0), -1)

    # Layout generation rendering via default fallback calls
    st.subheader("📷 Image Matrix Visualizations")
    st.image(thresh_img, caption="Binary Layer Slice", use_container_width=True)
    st.image(annotated_display, caption="Tagged Intersections", use_container_width=True)

    st.subheader("🎼 Mathematical Harmonic Output Matrix")
    if detected_tags:
        df_tags = pd.DataFrame(detected_tags).drop_duplicates(subset=["Harmonic Grid Link"])
        st.dataframe(df_tags, use_container_width=True)
    else:
        st.write("No matching harmonic junctions found at this layer combination.")
else:
    st.error("Data tracking array returned empty. The application cannot fetch data from the network.")
