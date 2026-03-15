import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title("Phases 1-5: Full Inscription Analysis")
st.write("Detect, vectorize, group, sequence, and analyze the grammar of symbols from an inscription.")

# --- Sidebar ---
st.sidebar.header("Tuning Parameters (Phase 1)")
canny_thresh1 = st.sidebar.slider("Canny Threshold 1", 10, 200, 60)
canny_thresh2 = st.sidebar.slider("Canny Threshold 2", 10, 400, 180)
blur_radius = st.sidebar.slider("Gaussian Blur Radius", 1, 11, 3, step=2)
min_contour_area = st.sidebar.slider("Min Symbol Area", 10, 500, 50)
max_contour_area = st.sidebar.slider("Max Symbol Area", 1000, 20000, 10000)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Use session state to hold data between button clicks
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

if uploaded_file is not None:
    # --- Run Analysis Button ---
    st.header("Run Full Analysis")
    selected_k = st.slider("Select the number of clusters (k) for Phase 3:", 10, 70, 30)
    
    if st.button(f"Run Full Analysis (Phases 1-5)"):
        with st.spinner("Performing full analysis... This may take a few moments."):
            # --- PHASES 1 & 2 ---
            img_color = Image.open(uploaded_file).convert("RGB")
            cv_img_gray = np.array(img_color.convert('L'))
            blurred = cv2.GaussianBlur(cv_img_gray, (blur_radius, blur_radius), 0)
            edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            valid_symbols = []
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                if min_contour_area < area < max_contour_area and w < 200 and h < 200:
                    moments = cv2.moments(contour)
                    hu_moments = cv2.HuMoments(moments)
                    log_transformed_hu = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))
                    
                    valid_symbols.append({
                        "vector": log_transformed_hu.flatten(), "x": x, "y": y
                    })
            
            st.session_state.symbol_count = len(valid_symbols)
            feature_vectors = np.array([s['vector'] for s in valid_symbols])
            
            # --- PHASE 3 ---
            kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init='auto').fit(feature_vectors)
            labels = kmeans.labels_
            for i, symbol in enumerate(valid_symbols):
                symbol['cluster'] = labels[i]
            
            # --- PHASE 4 ---
            valid_symbols.sort(key=lambda s: (s['y'] // 20, s['x']))
            final_sequence = [s['cluster'] for s in valid_symbols]
            
            # --- PHASE 5 ---
            num_clusters = selected_k
            transition_matrix = np.zeros((num_clusters, num_clusters))
            for i in range(len(final_sequence) - 1):
                from_state = final_sequence[i]
                to_state = final_sequence[i+1]
                transition_matrix[from_state, to_state] += 1
            
            # Normalize to get probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            # Avoid division by zero for states that are only at the end of a line
            prob_matrix = np.divide(transition_matrix, row_sums, out=np.zeros_like(transition_matrix), where=row_sums!=0)
            
            # Store results in session state
            st.session_state.prob_matrix = prob_matrix
            st.session_state.final_sequence_str = ' '.join(map(str, final_sequence))
            st.session_state.analysis_complete = True
            st.session_state.selected_k = selected_k

if st.session_state.analysis_complete:
    st.success("Full analysis is complete!")
    st.write("---")

    # --- Display Phase 4 Results ---
    st.header("Phase 4: Constructed Sequence")
    st.write(f"Found {st.session_state.symbol_count} symbols and constructed the digital transcript below:")
    st.code(st.session_state.final_sequence_str, language=None)
    st.write("---")

    # --- Display Phase 5 Results ---
    st.header("Phase 5: Grammar Discovery")
    st.write("The heatmap below shows the probability of one symbol following another. Bright squares indicate a high-probability pair, representing a likely grammatical or lexical rule.")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(st.session_state.prob_matrix, ax=ax, cmap="viridis")
    ax.set_title("Symbol Transition Probability Matrix")
    ax.set_xlabel("To Symbol (Cluster ID)")
    ax.set_ylabel("From Symbol (Cluster ID)")
    st.pyplot(fig)
    
    st.info("By examining the bright squares (e.g., from row 10 to column 25), you can identify the fundamental 'syllables' of the inscription's language.")
