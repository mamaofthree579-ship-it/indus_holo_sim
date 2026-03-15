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

st.title("Full Inscription Analysis (Phases 1-7)")
st.write("Detect, vectorize, group, sequence, analyze grammar, find patterns, and link glyphs to their meaning.")

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
    
    if st.button(f"Run Full Analysis"):
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
                    valid_symbols.append({
                        "patch": cv_img_gray[y:y+h, x:x+w],
                        "vector": log_transformed_hu.flatten(), 
                        "x": x, "y": y
                    })
            
            st.session_state.symbol_count = len(valid_symbols)
            feature_vectors = np.array([s['vector'] for s in valid_symbols])
            
            # --- PHASE 3 ---
            kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init='auto').fit(feature_vectors)
            for i, symbol in enumerate(valid_symbols):
                symbol['cluster'] = kmeans.labels_[i]
            
            # --- PHASE 4 ---
            valid_symbols.sort(key=lambda s: (s['y'] // 20, s['x']))
            final_sequence = [s['cluster'] for s in valid_symbols]
            
            # --- PHASE 5 & 6 ---
            bigrams = Counter(zip(final_sequence, final_sequence[1:]))
            trigrams = Counter(zip(final_sequence, final_sequence[1:], final_sequence[2:]))
            
            # --- Store results in session state ---
            st.session_state.valid_symbols = valid_symbols
            st.session_state.bigrams = bigrams
            st.session_state.trigrams = trigrams
            st.session_state.selected_k = selected_k
            st.session_state.analysis_complete = True

if st.session_state.analysis_complete:
    st.success("Full analysis is complete!")
    st.write("---")

    # --- PHASE 7: GLYPH DICTIONARY & SEMANTIC INFERENCE ---
    st.header("Phase 7: Glyph Dictionary & Semantic Inference")
    st.write("Use this dictionary to connect the statistical patterns to the actual symbol shapes.")

    # Create the visual dictionary
    glyph_dictionary = {}
    for symbol in st.session_state.valid_symbols:
        cluster_id = symbol['cluster']
        if cluster_id not in glyph_dictionary:
            glyph_dictionary[cluster_id] = symbol['patch']
            
    with st.expander("Show/Hide Glyph Dictionary"):
        cols = st.columns(10)
        for i in sorted(glyph_dictionary.keys()):
            col_index = i % 10
            with cols[col_index]:
                st.write(f"**ID: {i}**")
                st.image(glyph_dictionary[i], width=60)
    
    st.write("---")

    # Display Top Patterns for analysis
    col1, col2 = st.columns(2)
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
        
    st.write("---")
    
    # Guide the user's analysis
    st.subheader("Analysis Questions")
    st.markdown("""
    Now you can answer the key questions:
    1. **Examine the 'repeater' glyphs:** Look at the shapes for your top repeated bigrams (e.g., `6-6`, `5-5`, `8-8`). What do these glyphs look like?
    2. **Examine the 'central' glyph:** What does the shape of your most common symbol (likely **Glyph 6**) look like? Is it simple or complex?
    3. **Cross-reference with your data:** Now that you can see the shapes, what are your hypotheses about what they might mean?
    """)
