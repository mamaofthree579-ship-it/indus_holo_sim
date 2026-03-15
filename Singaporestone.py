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

st.title("The Complete Inscription Analysis Pipeline")
st.write("From raw image to final semantic interpretation. This tool will perform all 8 phases of the analysis.")

# --- Sidebar ---
st.sidebar.header("Phase 1: Tuning Parameters")
canny_thresh1 = st.sidebar.slider("Canny Threshold 1", 10, 200, 60)
canny_thresh2 = st.sidebar.slider("Canny Threshold 2", 10, 400, 180)
blur_radius = st.sidebar.slider("Gaussian Blur Radius", 1, 11, 3, step=2)
min_contour_area = st.sidebar.slider("Min Symbol Area", 10, 500, 50)
max_contour_area = st.sidebar.slider("Max Symbol Area", 1000, 20000, 10000)

uploaded_file = st.file_uploader("Choose your cleaned image file", type=["jpg", "jpeg", "png"])

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

if uploaded_file is not None:
    st.header("Run Full Analysis")
    selected_k = st.slider("Select the number of clusters (k) for Phase 3:", 10, 70, 30)
    
    if st.button(f"Run Full Analysis (Phases 1-8)"):
        with st.spinner("Performing end-to-end analysis... This is the final step."):
            # --- PHASES 1-4 ---
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
            
            feature_vectors = np.array([s['vector'] for s in valid_symbols])
            kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init='auto').fit(feature_vectors)
            for i, symbol in enumerate(valid_symbols):
                symbol['cluster'] = kmeans.labels_[i]
            
            valid_symbols.sort(key=lambda s: (s['y'] // 20, s['x']))
            final_sequence = [s['cluster'] for s in valid_symbols]
            
            # --- PHASES 5 & 6 ---
            bigrams = Counter(zip(final_sequence, final_sequence[1:]))
            trigrams = Counter(zip(final_sequence, final_sequence[2:]))
            quadrigrams = Counter(zip(final_sequence, final_sequence[3:]))
            
            # --- Store all results in session state ---
            st.session_state.valid_symbols = valid_symbols
            st.session_state.bigrams = bigrams
            st.session_state.trigrams = trigrams
            st.session_state.quadrigrams = quadrigrams
            st.session_state.analysis_complete = True

if st.session_state.analysis_complete:
    st.success("Analysis complete! The final results are below.")
    st.write("---")

    # --- PHASE 7: GLYPH DICTIONARY ---
    st.header("Phase 7: The Glyph Dictionary")
    glyph_dictionary = {s['cluster']: s['patch'] for s in reversed(st.session_state.valid_symbols)}
            
    with st.expander("Show/Hide Visual Glyph Dictionary"):
        cols = st.columns(10)
        for i in sorted(glyph_dictionary.keys()):
            with cols[i % 10]:
                st.write(f"**ID: {i}**")
                st.image(glyph_dictionary[i], width=60)
    
    st.write("---")

    # --- PHASE 8: FINAL SYNTHESIS & INTERPRETATION ---
    st.header("Phase 8: Final Synthesis - The Core Vocabulary")
    st.info("This section automatically displays the most frequent 'words' found in the inscription, showing both their statistical ID and their actual shape.")

    # Function to display a sequence
    def display_sequence(title, sequence_tuple, count):
        st.subheader(title)
        sequence_str = '-'.join(map(str, sequence_tuple))
        st.write(f"**Sequence:** `{sequence_str}` (Found {count} times)")
        
        cols = st.columns(len(sequence_tuple))
        for i, glyph_id in enumerate(sequence_tuple):
            with cols[i]:
                st.image(glyph_dictionary.get(glyph_id, Image.new('L', (50,50))), caption=f"ID: {glyph_id}", width=100)

    # Get the top patterns
    top_bigram, bigram_count = st.session_state.bigrams.most_common(1)[0]
    top_trigram, trigram_count = st.session_state.trigrams.most_common(1)[0]
    top_quadrigram, quadrigram_count = st.session_state.quadrigrams.most_common(1)[0]

    display_sequence("Most Common Bigram (2-Glyph Word)", top_bigram, bigram_count)
    st.write("---")
    display_sequence("Most Common Trigram (3-Glyph Word)", top_trigram, trigram_count)
    st.write("---")
    display_sequence("Most Common Quadrigram (4-Glyph Word)", top_quadrigram, quadrigram_count)
    st.write("---")

    st.header("Final Interpretation")
    st.success("Congratulations on completing the project!")
    st.markdown("""
    You have successfully built an algorithm that takes an unknown script from an image, identifies its characters, discovers its grammar, and extracts its most important words.

    The final step is human interpretation. Look at the visual patterns above and ask:
    - **What do the shapes suggest?** Are they objects, people, actions, or abstract concepts?
    - **How do the words build on each other?** Does the top bigram appear inside the top trigram?
    - **What story do they tell?** Based on the shapes and their frequency, what is the most likely subject of the inscription? A royal decree, a record of trade, a religious text, or something else entirely?

    This is the moment of discovery.
    """)
