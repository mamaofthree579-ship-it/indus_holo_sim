import streamlit as st
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from streamlit_drawable_canvas import st_canvas

# ==========================================
# HARMONIC MATRIX: JUST INTONATION FREQUENCIES
# ==========================================
JUST_INTONATION_SCALE = {
    "1:1 (Unison)": {"angle": 0.0, "freq": 220.0},
    "16:15 (Minor Second)": {"angle": 90.0 * np.log2(16/15), "freq": 234.6},
    "9:8 (Major Second)": {"angle": 90.0 * np.log2(9/8), "freq": 247.5},
    "6:5 (Minor Third)": {"angle": 90.0 * np.log2(6/5), "freq": 264.0},
    "5:4 (Major Third)": {"angle": 90.0 * np.log2(5/4), "freq": 275.0},
    "4:3 (Perfect Fourth)": {"angle": 90.0 * np.log2(4/3), "freq": 293.3},
    "45:32 (Tritone)": {"angle": 90.0 * np.log2(45/32), "freq": 309.4},
    "3:2 (Perfect Fifth)": {"angle": 90.0 * np.log2(3/2), "freq": 330.0},
    "8:5 (Minor Sixth)": {"angle": 90.0 * np.log2(8/5), "freq": 352.0},
    "5:3 (Major Sixth)": {"angle": 90.0 * np.log2(5/3), "freq": 366.6},
    "7:4 (Harmonic Seventh)": {"angle": 90.0 * np.log2(7/4), "freq": 385.0},
    "15:8 (Major Seventh)": {"angle": 90.0 * np.log2(15/8), "freq": 412.5},
    "2:1 (Octave / 90° Axis)": {"angle": 90.0, "freq": 440.0}
}

# --- Known Reference Matrix (Archetypes) ---
ARCHETYPES = {
    "Standard Fish Sign (Guild Identity A)": [0.0, 37.37, 52.65, 90.0],
    "U-Shape Pot Sign (Tax/Value Token)": [0.0, 28.97, 66.37, 90.0],
    "Parallel Line Bar (Commodity Measure)": [0.0, 90.0]
}

st.set_page_config(page_title="Indus Epigraphic Matrix Engine", layout="wide")
st.title("🏛️ Indus Epigraphic Multi-Layer Matrix Engine")
st.markdown("Advanced analytical workstation isolating structural layer tracking, audio sonification, and comparative matrix overlays.")

# --- Side Panel Controls ---
st.sidebar.header("🎛️ Analysis Configurations")
tolerance_deg = st.sidebar.slider("Harmonic Resonance Tolerance (Deg)", 0.5, 5.0, 2.0, step=0.5)

# --- Drawing Interface ---
st.subheader("✏️ Draw Layer Geometry Below")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=6,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=256,
    width=256,
    drawing_mode="freedraw",
    key="engine_canvas",
    return_image_data=True,
)

def process_binary_matrix(thresh, tol):
    """Core mathematical engine isolating angles and linking them to harmonic identities."""
    if thresh is None or np.sum(thresh) == 0:
        return [], None
    
    thresh = cv2.convertScaleAbs(thresh)
    sk = (skeletonize(thresh > 0).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(sk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    tags = []
    display_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
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
            angle_quad = np.degrees(np.arccos(cos_a)) % 90
            
            for label, data in JUST_INTONATION_SCALE.items():
                if abs(angle_quad - data["angle"]) <= tol:
                    tags.append({"label": label, "angle": angle_quad, "freq": data["freq"]})
                    cv2.circle(display_img, tuple(pts[i].astype(int)), 5, (0, 255, 255), -1)
                    break
    return tags, display_img

# --- Core Pipeline Execution ---
if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
    raw_canvas = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)
    
    # ----------------------------------------------------
    # 1. THE LAYER PERSISTENCE TRACKER (Multi-Depth Simulation)
    # ----------------------------------------------------
    st.header("⏳ 1. Structural Layer Persistence Tracker")
    st.markdown("Simulating micro-topography shifts by scanning across simulated physical engraving depths.")
    
    # Simulate variations across three simulated lighting depths
    simulated_depths = {"Surface Layer (High Light)": 50, "Median Engraving": 127, "Deep Core Groove": 200}
    persistence_rows = []
    active_frequencies = []
    
    cols = st.columns(3)
    for idx, (layer_name, simulated_thresh) in enumerate(simulated_depths.items()):
        # Introduce variation into each layer profile to simulate edge gradients
        layer_mask = cv2.GaussianBlur(raw_canvas, (3, 3), 0)
        _, layer_thresh = cv2.threshold(layer_mask, simulated_thresh, 255, cv2.THRESH_BINARY)
        
        tags, img_out = process_binary_matrix(layer_thresh, tolerance_deg)
        
        with cols[idx]:
            if img_out is not None:
                st.image(img_out, caption=layer_name, use_container_width=True)
            else:
                st.caption(f"{layer_name} empty")
                
        found_labels = set([t["label"] for t in tags]) if tags else set()
        for label in JUST_INTONATION_SCALE.keys():
            persistence_rows.append({
                "Harmonic Scale Tier": label,
                "Layer Depth": layer_name,
                "Status": "🔒 LOCKED" if label in found_labels else "░ Open"
            })
        if tags:
            for t in tags:
                if t["freq"] not in active_frequencies:
                    active_frequencies.append(t["freq"])

    df_p = pd.DataFrame(persistence_rows)
    df_pivot = df_p.pivot(index="Harmonic Scale Tier", columns="Layer Depth", values="Status")
    st.dataframe(df_pivot, use_container_width=True)

    # ----------------------------------------------------
    # 2. AUDIO SYNTHESIZER NODE (Sonification Engine)
    # ----------------------------------------------------
    st.header("🎼 2. Acoustic Sonification Chord Node")
    st.markdown("Converts the active geometric grid lines into a real-time audio tone matrix.")
    
    if active_frequencies:
        # Sort and take the top unique frequencies to keep the chord clear
        chord_freqs = sorted(list(set(active_frequencies)))[:4]
        st.write(f"**Calculated Geometric Acoustic Pitch Vectors:** {', '.join([f'{f}Hz' for f in chord_freqs])}")
        
        # Inject an inline Web Audio synthesizer engine script
        oscillator_code = "".join([f"""
            let osc{i} = audioCtx.createOscillator();
            let gain{i} = audioCtx.createGain();
            osc{i}.type = 'sine';
            osc{i}.frequency.setValueAtTime({freq}, audioCtx.currentTime);
            gain{i}.gain.setValueAtTime(0.15, audioCtx.currentTime);
            osc{i}.connect(gain{i});
            gain{i}.connect(audioCtx.destination);
            osc{i}.start();
            osc{i}.stop(audioCtx.currentTime + 2.5);
        """ for i, freq in enumerate(chord_freqs)])
        
        audio_trigger_script = f"""
            <button onclick="playHarmonicMatrixChord()" style="
                background-color: #00FFCC; color: #000; border: none; 
                padding: 12px 24px; font-weight: bold; border-radius: 4px; cursor: pointer;">
                🔊 Play Geometric Grid Resonance Chord
            </button>
            <script>
            function playHarmonicMatrixChord() {{
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                const audioCtx = new AudioContext();
                {oscillator_code}
            }}
            </script>
        """
        st.components.v1.html(audio_trigger_script, height=60)
    else:
        st.info("Draw intersecting angles on the canvas above to build an acoustic chord matrix.")

    # ----------------------------------------------------
    # 3. MATRIX COMPARE TOOL (System Validation)
    # ----------------------------------------------------
    st.header("⚖️ 3. Macro Matrix Archetype Comparison")
    st.markdown("Evaluates whether your drawn shape signature aligns with known administrative seal templates.")
    
    # Process current drawing data into an angular signature list
    current_tags, _ = process_binary_matrix(raw_canvas, tolerance_deg)
    current_angles = sorted(list(set([t["angle"] for t in current_tags]))) if current_tags else []
    
    if current_angles:
        comparison_results = []
        for arch_name, arch_angles in ARCHETYPES.items():
            # Calculate the structural distance matrix between coordinate groups
            arr_drawn = np.array(current_angles).reshape(-1, 1)
            arr_arch = np.array(arch_angles).reshape(-1, 1)
            dist_matrix = cdist(arr_drawn, arr_arch, metric='euclidean')
            structural_deviation = np.mean(np.min(dist_matrix, axis=1))
            
            # Translate deviation error into a percentage match
            match_confidence = max(0.0, 100.0 - (structural_deviation * 2.5))
            comparison_results.append({
                "System Archetype Registry": arch_name,
                "Structural Deviation Index": round(structural_deviation, 3),
                "Matrix Match Confidence": f"{round(match_confidence, 1)}%"
            })
            
        df_comp = pd.DataFrame(comparison_results).sort_values(by="Structural Deviation Index")
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.caption("Awaiting distinct angular intersections to run database cross-checking loops.")
else:
    st.info("🖋️ Awaiting vector inputs. Use your mouse to draw a symbol on the canvas above to initialize the three analytical frameworks.")
