import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

st.set_page_config(layout="wide")

st.title("Phases 1-6: Full Inscription Analysis")
st.write("Detect, vectorize, group, sequence, analyze grammar, and find recurring patterns in an inscription.")

# --- Sidebar ---
st.sidebar.header("Tuning Parameters (Phase 1)")
canny_thresh1 = st.sidebar.slider("Canny Threshold 1", 10, 200, 60)
canny_thresh2 = st.sidebar.slider("Canny Threshold 2", 10, 400, 180)
blur_radius = st.sidebar.slider("Gaussian Blur Radius", 1, 11, 3, step=2)
min_contour_area = st.sidebar.slider("Min Symbol Area", 10, 500, 50)
max_contour_area = st.sidebar.slider("Max Symbol Area", 1000, 20000, 10000)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

if uploaded_file is not None:
    st.header("Run Full Analysis")
    selected_k = st.slider("Select the number of clusters (k) for Phase 3:", 10, 70, 30)
    
    if st.button(f"Run Full Analysis (Phases 1-6)"):
        with st.spinner("Performing full analysis... This will take a moment."):
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
                    valid_symbols.append({"vector": log_transformed_hu.flatten(), "x": x, "y": y})
            
            st.session_state.symbol_count = len(valid_symbols)
            feature_vectors = np.array([s['vector'] for s in valid_symbols])
            
            # --- PHASE 3 ---
            kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init='auto').fit(feature_vectors)
            for i, symbol in enumerate(valid_symbols):
                symbol['cluster'] = kmeans.labels_[i]
            
            # --- PHASE 4 ---
            valid_symbols.sort(key=lambda s: (s['y'] // 20, s['x']))
            final_sequence = [s['cluster'] for s in valid_symbols]
            
            # --- PHASE 5 ---
            prob_matrix = np.zeros((selected_k, selected_k))
            for i in range(len(final_sequence) - 1):
                prob_matrix[final_sequence[i], final_sequence[i+1]] += 1
            row_sums = prob_matrix.sum(axis=1, keepdims=True)
            prob_matrix = np.divide(prob_matrix, row_sums, out=np.zeros_like(prob_matrix), where=row_sums!=0)
            
            # --- PHASE 6 ---
            def find_ngrams(sequence, n):
                ngrams = zip(*[sequence[i:] for i in range(n)])
                return Counter(ngrams)

            st.session_state.bigrams = find_ngrams(final_sequence, 2)
            st.session_state.trigrams = find_ngrams(final_sequence, 3)
            st.session_state.quadrigrams = find_ngrams(final_sequence, 4)
            
            # Store other results
            st.session_state.prob_matrix = prob_matrix
            st.session_state.final_sequence_str = ' '.join(map(str, final_sequence))
            st.session_state.analysis_complete = True

if st.session_state.analysis_complete:
    st.success("Full analysis is complete!")
    st.write("---")

    # Display Phase 5 Results
    st.header("Phase 5: Grammar Discovery")
    st.write("The heatmap shows the probability of one symbol following another.")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(st.session_state.prob_matrix, ax=ax, cmap="viridis")
    st.pyplot(fig)
    st.write("---")

    # Display Phase 6 Results
    st.header("Phase 6: Recursive Pattern Detection")
    st.write("These tables show the most frequently occurring symbol sequences (words and phrases) found in the text.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Top Bigrams (2-symbol words)")
        bigram_df = pd.DataFrame(st.session_state.bigrams.most_common(10), columns=['Sequence', 'Count'])
        bigram_df['Sequence'] = bigram_df['Sequence'].apply(lambda x: f"{x[0]}-{x[1]}")
        st.dataframe(bigram_df)
    with col2:
        st.subheader("Top Trigrams (3-symbol words)")
        trigram_df = pd.DataFrame(st.session_state.trigrams.most_common(10), columns=['Sequence', 'Count'])
        trigram_df['Sequence'] = trigram_df['Sequence'].apply(lambda x: f"{x[0]}-{x[1]}-{x[2]}")
        st.dataframe(trigram_df)
    with col3:
        st.subheader("Top Quadrigrams (4-symbol words)")
        quadrigram_df = pd.DataFrame(st.session_state.quadrigrams.most_common(10), columns=['Sequence', 'Count'])
        quadrigram_df['Sequence'] = quadrigram_df['Sequence'].apply(lambda x: f"{x[0]}-{x[1]}-{x[2]}-{x[3]}")
        st.dataframe(quadrigram_df)
        
    st.info("Look for patterns that build on each other. For example, if '20-19' is a top bigram, is '20-19-10' a top trigram?")
