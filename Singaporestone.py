import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2

st.title("Phase 1: Inscription Symbol Extraction")
st.write("Upload an image of an artifact to detect and isolate potential symbols or glyphs.")

# --- Sidebar for tunable parameters ---
st.sidebar.header("Tuning Parameters")
st.sidebar.write("Adjust these values to refine the symbol detection.")

canny_thresh1 = st.sidebar.slider("Canny Threshold 1", min_value=10, max_value=200, value=60)
canny_thresh2 = st.sidebar.slider("Canny Threshold 2", min_value=10, max_value=400, value=180)
blur_radius = st.sidebar.slider("Gaussian Blur Radius", min_value=1, max_value=11, value=3, step=2)
min_contour_area = st.sidebar.slider("Min Symbol Area", min_value=10, max_value=500, value=50)
max_contour_area = st.sidebar.slider("Max Symbol Area", min_value=1000, max_value=20000, value=10000)

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Load Image ---
    try:
        img_color = Image.open(uploaded_file).convert("RGB")
        st.image(img_color, caption="Original Uploaded Image", use_column_width=True)
        st.write("---")
        st.header("Analysis Results")
        
        # --- Phase 1 Processing ---
        
        # 1. Convert to Grayscale
        cv_img_gray = np.array(img_color.convert('L'))

        # 2. Reduce Noise (Gaussian Blur)
        blurred = cv2.GaussianBlur(cv_img_gray, (blur_radius, blur_radius), 0)

        # 3. Detect Edges (Canny)
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

        # 4. Identify Contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Filter and Draw Bounding Boxes
        img_with_boxes = img_color.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        symbol_count = 0

        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            
            # Filter contours based on area and simple aspect ratio to remove noise
            if min_contour_area < area < max_contour_area and w < 200 and h < 200:
                draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
                symbol_count += 1
        
        # --- Display Results ---
        st.image(img_with_boxes, caption=f"Detected Symbols: {symbol_count}", use_column_width=True)
        st.success(f"Phase 1 Complete: Identified {symbol_count} potential symbols.")
        st.info("The red boxes highlight the detected symbols based on the current parameters. Adjust the sliders in the sidebar to improve detection.")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

else:
    st.info("Please upload an image to begin the analysis.")
