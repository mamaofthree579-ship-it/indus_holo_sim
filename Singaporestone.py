import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Phases 1-4: Full Inscription Analysis")
st.write("Detect symbols, vectorize them, group them into families, and construct the final reading sequence.")

# --- Sidebar ---
st.sidebar.header("Tuning Parameters (Phase 1)")
canny_thresh1 = st.sidebar.slider("Canny Threshold 1", 10, 200, 60)
canny_thresh2 = st.sidebar.slider("Canny Threshold 2", 10, 400, 180)
blur_radius = st.sidebar.slider("Gaussian Blur Radius", 1, 11, 3, step=2)
min_contour_area = st.sidebar.slider("Min Symbol Area", 10, 500, 50)
max_contour_area = st.sidebar.slider("Max Symbol Area", 1000, 20000, 10000)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if 'phase_4_complete' not in st.session_state:
    st.session_state.phase_4_complete = False

if uploaded_file is not None:
    try:
        # --- PHASE 1 & 2 ---
        img_color = Image.open(uploaded_file).convert("RGB")
        cv_img_gray = np.array(img_color.convert('L'))
        blurred = cv2.GaussianBlur(cv_img_gray, (blur_radius, blur_radius), 0)
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_with_boxes = img_color.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        valid_symbols = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if min_contour_area < area < max_contour_area and w < 200 and h < 200:
                moments = cv2.moments(contour)
                hu_moments = cv2.HuMoments(moments)
                log_transformed_hu = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))
                
                valid_symbols.append({
                    "patch": cv_img_gray[y:y+h, x:x+w],
                    "vector": log_transformed_hu.flatten(),
                    "x": x, "y": y
                })
                draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        
        symbol_count = len(valid_symbols)
        st.header("Phase 1 & 2 Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_with_boxes, caption=f"Phase 1: Detected {symbol_count} symbols", use_column_width=True)
        with col2:
            if symbol_count > 0:
                st.write(f"Phase 2: Generated {symbol_count} feature vectors.")
        st.write("---")

        # --- PHASE 3 ---
        st.header("Phase 3: Symbol Classification")
        if symbol_count > 10:
            feature_vectors = np.array([s['vector'] for s in valid_symbols])
            
            with st.expander("Click to see Elbow Plot for choosing 'k'"):
                k_range = range(10, 71, 5)
                inertias = [KMeans(n_clusters=k, random_state=42, n_init='auto').fit(feature_vectors).inertia_ for k in k_range]
                fig, ax = plt.subplots()
                ax.plot(k_range, inertias, marker='o')
                ax.set_xlabel("Number of clusters (k)")
                ax.set_ylabel("Inertia")
                ax.set_title("Elbow Method for Optimal k")
                st.pyplot(fig)

            selected_k = st.slider("Select the number of clusters (k):", 10, 70, 30)
            
            if st.button(f"Group Symbols and Construct Sequence"):
                st.session_state.phase_4_complete = True
                kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init='auto').fit(feature_vectors)
                labels = kmeans.labels_
                for i, symbol in enumerate(valid_symbols):
                    symbol['cluster'] = labels[i]
                
                st.session_state.valid_symbols = valid_symbols
                st.session_state.selected_k = selected_k

        if st.session_state.phase_4_complete:
            # --- PHASE 4: SEQUENCE CONSTRUCTION ---
            st.write("---")
            st.header("Phase 4: Sequence Construction")
            
            with st.spinner("Sorting symbols into reading order..."):
                symbols_to_sort = st.session_state.valid_symbols
                
                # Sort primarily by 'y' coordinate, then by 'x'
                # A more robust line-finding algorithm can be complex, but this is a strong start
                symbols_to_sort.sort(key=lambda s: (s['y'] // 20, s['x'])) # Group by y-lines of 20px height
                
                final_sequence = [s['cluster'] for s in symbols_to_sort]
                
                st.success("Successfully constructed the symbol sequence!")
                st.write("This is the 'digital transcript' of the inscription, where each number represents a glyph family:")
                
                # Display the sequence as a string of numbers
                sequence_str = ' '.join(map(str, final_sequence))
                st.code(sequence_str, language=None)
                
                # Visualize the first part of the sequence
                st.subheader("Visual Transcript (first 50 symbols)")
                
                # Create a mapping from cluster ID to a sample image
                cluster_to_patch = {i: None for i in range(st.session_state.selected_k)}
                for s in symbols_to_sort:
                    if cluster_to_patch[s['cluster']] is None:
                        cluster_to_patch[s['cluster']] = s['patch']
                
                # Display images in order
                cols = st.columns(10)
                for i in range(min(50, len(final_sequence))):
                    cluster_id = final_sequence[i]
                    patch = cluster_to_patch.get(cluster_id)
                    if patch is not None:
                        with cols[i % 10]:
                            st.image(patch, width=40, caption=f"ID:{cluster_id}")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
