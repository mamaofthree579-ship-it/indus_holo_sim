import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter

st.set_page_config(layout="wide")

st.title("The Intelligent Inscription Analysis Pipeline")
st.info("This version includes advanced shape filtering to distinguish between glyphs and noise (scratches, stains).")

# --- Sidebar ---
st.sidebar.header("Phase 1: Detection & Filtering")
st.sidebar.subheader("Edge & Size Filters")
canny_thresh1 = st.sidebar.slider("Canny Threshold 1", 10, 200, 60)
canny_thresh2 = st.sidebar.slider("Canny Threshold 2", 10, 400, 180)
min_contour_area = st.sidebar.slider("Min Symbol Area", 10, 1000, 100)
max_contour_area = st.sidebar.slider("Max Symbol Area", 1000, 20000, 10000)

st.sidebar.subheader("Shape Filtering")
min_solidity = st.sidebar.slider("Minimum Solidity (0.0 to 1.0)", 0.0, 1.0, 0.35, 0.05)
st.sidebar.markdown("_High values (e.g., > 0.8) keep only solid, filled-in shapes. Low values (< 0.2) allow thin, scratch-like shapes. Adjust this to filter out noise._")

uploaded_file = st.file_uploader("Choose your image file", type=["jpg", "jpeg", "png"])

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

if uploaded_file is not None:
    st.header("Run Full Analysis")
    selected_k = st.slider("Select the number of clusters (k) for Phase 3:", 10, 70, 30)
    
    if st.button(f"Run Full Intelligent Analysis"):
        with st.spinner("Applying intelligent filters and running full analysis..."):
            # --- PHASE 1: Intelligent Detection ---
            img_color = Image.open(uploaded_file).convert("RGB")
            cv_img_gray = np.array(img_color.convert('L'))
            
            # Use a blur to help smooth noise before detection
            blurred = cv2.GaussianBlur(cv_img_gray, (5, 5), 0)
            
            edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_symbols = []
            img_with_contours = img_color.copy() # For visualization

            for contour in contours:
                area = cv2.contourArea(contour)
                
                # --- NEW SHAPE FILTERING LOGIC ---
                if area > 0:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                else:
                    solidity = 0

                if min_contour_area < area < max_contour_area and solidity > min_solidity:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Additional filter for aspect ratio can be added here if needed
                    # if w > 5 and h > 5:
                    
                    moments = cv2.moments(contour)
                    hu_moments = cv2.HuMoments(moments)
                    log_transformed_hu = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))
                    
                    valid_symbols.append({
                        "patch": cv_img_gray[y:y+h, x:x+w],
                        "vector": log_transformed_hu.flatten(), 
                        "x": x, "y": y
                    })
                    # Draw a green box around accepted glyphs
                    cv2.rectangle(img_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            st.session_state.img_with_contours = img_with_contours
            st.session_state.detected_count = len(valid_symbols)

            if len(valid_symbols) < selected_k:
                st.error(f"Analysis failed: Only {len(valid_symbols)} valid glyphs were detected, which is fewer than the requested {selected_k} clusters. Please adjust your filter settings (especially try lowering 'Minimum Solidity') and try again.")
            else:
                # --- PHASES 2-8 ---
                feature_vectors = np.array([s['vector'] for s in valid_symbols])
                kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init='auto').fit(feature_vectors)
                for i, symbol in enumerate(valid_symbols):
                    symbol['cluster'] = kmeans.labels_[i]
                
                valid_symbols.sort(key=lambda s: (s['y'] // 20, s['x']))
                final_sequence = [s['cluster'] for s in valid_symbols]
                
                bigrams = Counter(zip(final_sequence, final_sequence[1:]))
                trigrams = Counter(zip(final_sequence, final_sequence[2:]))
                quadrigrams = Counter(zip(final_sequence, final_sequence[3:]))
                
                st.session_state.valid_symbols = valid_symbols
                st.session_state.bigrams = bigrams
                st.session_state.trigrams = trigrams
                st.session_state.quadrigrams = quadrigrams
                st.session_state.analysis_complete = True

if st.session_state.analysis_complete:
    st.success("Analysis complete!")

    st.header("Phase 1 Report: Detected Glyphs")
    st.write(f"The algorithm identified **{st.session_state.detected_count}** shapes that meet your criteria. The green boxes show the accepted glyphs.")
    st.image(st.session_state.img_with_contours, caption="Green boxes show what was analyzed.")
    st.write("---")

    # --- Phase 7 & 8 ---
    st.header("Core Vocabulary & Interpretation")
    glyph_dictionary = {s['cluster']: s['patch'] for s in reversed(st.session_state.valid_symbols)}

    def display_sequence(title, sequence_tuple, count):
        st.subheader(title)
        sequence_str = '-'.join(map(str, sequence_tuple))
        st.write(f"**Sequence:** `{sequence_str}` (Found {count} times)")
        cols = st.columns(len(sequence_tuple))
        for i, glyph_id in enumerate(sequence_tuple):
            with cols[i]:
                st.image(glyph_dictionary.get(glyph_id, Image.new('L', (50,50))), caption=f"ID: {glyph_id}", width=100)

    # Check if any patterns were found
    if st.session_state.bigrams:
        top_bigram, bigram_count = st.session_state.bigrams.most_common(1)[0]
        display_sequence("Most Common Bigram (2-Glyph Word)", top_bigram, bigram_count)
    else:
        st.warning("No repeating bigrams were found with the current settings.")
    
    st.write("---")
    
    if st.session_state.trigrams:
        top_trigram, trigram_count = st.session_state.trigrams.most_common(1)[0]
        display_sequence("Most Common Trigram (3-Glyph Word)", top_trigram, trigram_count)
    else:
        st.warning("No repeating trigrams were found with the current settings.")

    st.write("---")
    st.header("Project Conclusion")
    st.success("This application is now a complete, intelligent tool for basic computational archaeology. You can tune the filters in the sidebar and re-run the analysis on any image to rapidly test hypotheses about its content.")
