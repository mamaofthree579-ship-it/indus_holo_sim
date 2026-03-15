import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Phases 1-3: Symbol Extraction, Vectorization, and Clustering")
st.write("Upload an image to detect symbols, convert them to vectors, and group them into families using k-Means clustering.")

# --- Sidebar for tunable parameters ---
st.sidebar.header("Tuning Parameters (Phase 1)")
canny_thresh1 = st.sidebar.slider("Canny Threshold 1", 10, 200, 60)
canny_thresh2 = st.sidebar.slider("Canny Threshold 2", 10, 400, 180)
blur_radius = st.sidebar.slider("Gaussian Blur Radius", 1, 11, 3, step=2)
min_contour_area = st.sidebar.slider("Min Symbol Area", 10, 500, 50)
max_contour_area = st.sidebar.slider("Max Symbol Area", 1000, 20000, 10000)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img_color = Image.open(uploaded_file).convert("RGB")
        
        # --- PHASE 1 & 2 ---
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
                # Store the contour and its bounding box
                symbol_patch = cv_img_gray[y:y+h, x:x+w]
                moments = cv2.moments(contour)
                hu_moments = cv2.HuMoments(moments)
                log_transformed_hu = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))
                
                valid_symbols.append({
                    "contour": contour,
                    "patch": symbol_patch,
                    "vector": log_transformed_hu.flatten()
                })
                draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        
        symbol_count = len(valid_symbols)

        # Display Phase 1 and 2 results
        st.header("Phase 1 & 2 Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_with_boxes, caption=f"Phase 1: Detected {symbol_count} symbols", use_column_width=True)
        
        with col2:
            if symbol_count > 0:
                feature_vectors = [s['vector'] for s in valid_symbols]
                df_vectors = pd.DataFrame(feature_vectors, columns=[f"Hu_{i+1}" for i in range(7)])
                st.write(f"Phase 2: Generated {len(df_vectors)} feature vectors.")
                st.dataframe(df_vectors.head())
        
        st.write("---")

        # --- PHASE 3: SYMBOL CLASSIFICATION ---
        st.header("Phase 3: Symbol Classification")
        if symbol_count > 10: # Need enough symbols to cluster
            feature_vectors = np.array([s['vector'] for s in valid_symbols])

            # 1. Elbow Method to find optimal K
            st.subheader("Step 3.1: Find the Optimal Number of Clusters (k)")
            with st.spinner("Calculating the elbow plot... This may take a moment."):
                k_range = range(10, 71, 5) # Check k from 10 to 70
                inertias = []
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    kmeans.fit(feature_vectors)
                    inertias.append(kmeans.inertia_)
                
                fig, ax = plt.subplots()
                ax.plot(k_range, inertias, marker='o')
                ax.set_xlabel("Number of clusters (k)")
                ax.set_ylabel("Inertia")
                ax.set_title("Elbow Method for Optimal k")
                st.pyplot(fig)
            st.info("Look for the 'elbow' in the plot above – the point where the curve starts to flatten. This point suggests the best number of clusters for the data.")

            # 2. User selects K and runs K-Means
            st.subheader("Step 3.2: Run Clustering")
            selected_k = st.slider("Select the number of clusters (k) based on the plot:", min_value=10, max_value=70, value=30)

            if st.button(f"Group Symbols into {selected_k} Families"):
                with st.spinner(f"Running k-Means with k={selected_k}..."):
                    kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init='auto')
                    kmeans.fit(feature_vectors)
                    labels = kmeans.labels_

                    # Append labels to our symbols data
                    for i, symbol in enumerate(valid_symbols):
                        symbol['cluster'] = labels[i]

                    st.success(f"Successfully classified symbols into {selected_k} families.")
                    
                    # 3. Visualize the clusters
                    st.subheader("Discovered Glyph Families")
                    
                    # Display a sample from each cluster
                    for i in range(selected_k):
                        st.write(f"**Cluster {i+1}:**")
                        cluster_symbols = [s['patch'] for s in valid_symbols if s['cluster'] == i]
                        
                        # Use columns for a neater grid layout
                        cols = st.columns(10) 
                        for j in range(min(len(cluster_symbols), 10)):
                            with cols[j]:
                                st.image(cluster_symbols[j], width=50, caption=f"Symbol {j+1}")

        else:
            st.warning("Not enough symbols detected to perform clustering.")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
