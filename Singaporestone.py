import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pandas as pd

st.set_page_config(layout="wide") # Use a wider layout for better display

st.title("Phases 1 & 2: Symbol Extraction and Vectorization")
st.write("Upload an image to detect symbols (Phase 1) and convert them into numerical feature vectors (Phase 2).")

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
    try:
        img_color = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.header("Original Image")
            st.image(img_color, use_column_width=True)
            
        # --- PHASE 1: IMAGE SYMBOL EXTRACTION ---
        cv_img_gray = np.array(img_color.convert('L'))
        blurred = cv2.GaussianBlur(cv_img_gray, (blur_radius, blur_radius), 0)
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_with_boxes = img_color.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if min_contour_area < area < max_contour_area and w < 200 and h < 200:
                valid_contours.append(c)
                draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        
        symbol_count = len(valid_contours)
        
        with col2:
            st.header("Phase 1: Symbol Detection")
            st.image(img_with_boxes, use_column_width=True)
            st.success(f"Identified {symbol_count} potential symbols.")
        
        st.write("---")

        # --- PHASE 2: SYMBOL VECTORIZATION ---
        st.header("Phase 2: Symbol Vectorization Results")
        if symbol_count > 0:
            feature_vectors = []
            for contour in valid_contours:
                # Calculate moments for each contour
                moments = cv2.moments(contour)
                # Calculate Hu Moments from the moments
                hu_moments = cv2.HuMoments(moments)
                # Log transform Hu Moments to make them more stable for clustering
                # This is a standard practice as the raw values can vary by many orders of magnitude
                log_transformed_hu = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))
                # Flatten the array to a simple list for our dataset
                feature_vectors.append(log_transformed_hu.flatten())

            # Display the results in a DataFrame
            df_vectors = pd.DataFrame(feature_vectors, columns=[f"Hu_{i+1}" for i in range(7)])
            st.success(f"Successfully generated {len(df_vectors)} feature vectors.")
            st.write("Each row represents a symbol, and each column is a geometric feature (a Hu Moment).")
            st.dataframe(df_vectors)
        else:
            st.warning("No symbols were detected in Phase 1, so no vectors could be generated.")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

else:
    st.info("Please upload an image to begin the analysis.")
