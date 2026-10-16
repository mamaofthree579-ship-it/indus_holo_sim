import streamlit as st
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from streamlit_drawable_canvas import st_canvas

# ========================================================
# HARMONIC MATRIX: JUST INTONATION CONSTANTS
# ========================================================
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

# --- System Validation Template Repositories ---
ARCHETYPES = {
    "Standard Fish Sign (Guild Identity A)": [0.0, 37.37, 52.65, 90.0],
    "U-Shape Pot Sign (Tax/Value Token)": [0.0, 28.97, 66.37, 90.0],
    "Parallel Line Bar (Commodity Measure)": [0.0, 90.0]
}

# --- Streamlit Dashboard Initializing ---
st.set_page_config(page_title="Indus Epigraphic Matrix Engine", layout="wide")
st.title("🏛️ Indus Epigraphic Multi-Layer Matrix Engine")
st.markdown("Advanced analytical workstation isolating structural layer tracking, audio sonification, and authenticity fingerprints.")

st.sidebar.header("🎛️ Analysis Configurations")
tolerance_deg = st.sidebar.slider("Harmonic Resonance Tolerance (Deg)", 0.5, 5.0, 2.0, step=0.5)

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
    """Core geometric filtering engine parsing joint angle configurations."""
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
                    tags.append({
                        "Coordinate Location": str(tuple(pts[i].astype(int))), 
                        "Measured Axis Angle": f"{round(angle_quad, 1)}°", 
                        "Harmonic Grid Link": label, 
                        "angle": angle_quad, 
                        "freq": data["freq"]
                    })
                    cv2.circle(display_img, tuple(pts[i].astype(int)), 5, (0, 255, 255), -1)
                    break
    return tags, display_img
# --- Core Pipeline Execution Loop ---
if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
    # Safely convert the drawing into a grayscale canvas right away
    user_canvas = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)
    
    # ----------------------------------------------------
    # NEW: CLAY IMPRESSION SIMULATION ENGINE
    # ----------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("🏺 Matrix View Mode")
    view_mode = st.sidebar.radio(
        "Select Physical Projection:",
        ["Original Seal View (Carved Intaglio)", "Clay Impression View (Stamped Relief)"]
    )
    
    # If Clay mode is active, horizontally mirror the coordinate matrix (Axis 1)
    if view_mode == "Clay Impression View (Stamped Relief)":
        raw_canvas = cv2.flip(user_canvas, 1)
        st.sidebar.success("🔄 Horizontal Mirror Active: Simulating Clay Compression.")
    else:
        raw_canvas = user_canvas

    _, thresh_img = cv2.threshold(raw_canvas, 127, 255, cv2.THRESH_BINARY)
    
    # ----------------------------------------------------
    # AUTOMATED FINGERPRINT MATRIX SWEEP
    # ----------------------------------------------------
    st.header("🧬 Automated Authenticity Fingerprint Sweep")
    st.markdown("Scanning 10 layers simultaneously to compute spatial stability indexes.")
    
    sweep_thresholds = np.linspace(30, 230, 10, dtype=int)
    harmonic_counts = {label: 0 for label in JUST_INTONATION_SCALE.keys()}
    total_valid_sweeps = 0
    
    for t_val in sweep_thresholds:
        layer_mask = cv2.GaussianBlur(raw_canvas, (3, 3), 0)
        _, layer_thresh = cv2.threshold(layer_mask, t_val, 255, cv2.THRESH_BINARY)
        tags, _ = process_binary_matrix(layer_thresh, tolerance_deg)
        
        if tags:
            total_valid_sweeps += 1
            found_this_sweep = set([t["Harmonic Grid Link"] for t in tags])
            for label in found_this_sweep:
                harmonic_counts[label] += 1
                
    fingerprint_data = []
    high_stability_intervals = 0
    
    for label, count in harmonic_counts.items():
        persistence_pct = (count / total_valid_sweeps * 100) if total_valid_sweeps > 0 else 0.0
        if persistence_pct >= 70.0:
            high_stability_intervals += 1
            
        fingerprint_data.append({
            "Harmonic Data Tag": label,
            "Layer Persistence Frequency": f"{count} / {total_valid_sweeps} Layers",
            "Stability Percentage": persistence_pct
        })
        
    base_score = (high_stability_intervals / 4.0) * 100.0  
    authenticity_score = min(100.0, max(0.0, base_score))
    
    score_col1, score_col2 = st.columns(2)
    with score_col1:
        st.metric(
            label="🔒 Authenticity Fingerprint Score", 
            value=f"{round(authenticity_score, 1)}%",
            help="High percentages signal structural design resistance confirming systematic intentional layouts."
        )
    with score_col2:
        if authenticity_score >= 75.0:
            st.success("🎯 **VERDICT: Cryptographic Token Archetype.** Geometry demonstrates complete resistance to distortion, confirming a structured blueprint.")
        elif authenticity_score >= 40.0:
            st.warning("⚠️ **VERDICT: Indeterminate Layering.** Signs show partial grid distortion typical of secondary hybrid components.")
        else:
            st.error("❌ **VERDICT: High Distortion / Freehand Artwork.** Geometric relationships vanish rapidly across physical depth slices.")

    # ----------------------------------------------------
    # 1. LAYER PERSISTENCE DISPLAY
    # ----------------------------------------------------
    st.header("⏳ 1. Structural Layer Persistence Tracker")
    simulated_depths = {"Surface Layer": 50, "Median Engraving": 127, "Deep Core Groove": 200}
    persistence_rows = []
    active_frequencies = []
    
    cols = st.columns(3)
    for idx, (layer_name, simulated_thresh) in enumerate(simulated_depths.items()):
        layer_mask = cv2.GaussianBlur(raw_canvas, (3, 3), 0)
        _, layer_thresh = cv2.threshold(layer_mask, simulated_thresh, 255, cv2.THRESH_BINARY)
        tags, img_out = process_binary_matrix(layer_thresh, tolerance_deg)
        
        with cols[idx]:
            if img_out is not None:
                st.image(img_out, caption=f"{layer_name} ({view_mode})", use_container_width=True)
                
        found_labels = set([t["Harmonic Grid Link"] for t in tags]) if tags else set()
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
    # 2. AUDIO CHORD SYNTHESIZER
    # ----------------------------------------------------
    st.header("🎼 2. Acoustic Sonification Chord Node")
    if active_frequencies:
        chord_freqs = sorted(list(set(active_frequencies)))[:4]
        st.write(f"**Acoustic Vector Footprint Matrix:** {', '.join([f'{f}Hz' for f in chord_freqs])}")
        
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

    # ----------------------------------------------------
    # 3. OVERLAY COMPARATOR INDEX
    # ----------------------------------------------------
    st.header("⚖️ 3. Macro Matrix Archetype Comparison")
    current_tags, _ = process_binary_matrix(thresh_img, tolerance_deg)
    current_angles = sorted(list(set([t["angle"] for t in current_tags]))) if current_tags else []
    
    if current_angles:
        comparison_results = []
        for arch_name, arch_angles in ARCHETYPES.items():
            arr_drawn = np.array(current_angles).reshape(-1, 1)
            arr_arch = np.array(arch_angles).reshape(-1, 1)
            dist_matrix = cdist(arr_drawn, arr_arch, metric='euclidean')
            structural_deviation = np.mean(np.min(dist_matrix, axis=1))
            
            match_confidence = max(0.0, 100.0 - (structural_deviation * 2.5))
            comparison_results.append({
                "System Archetype Registry": arch_name,
                "Structural Deviation Index": round(structural_deviation, 3),
                "Matrix Match Confidence": f"{round(match_confidence, 1)}%"
            })
        st.dataframe(pd.DataFrame(comparison_results).sort_values(by="Structural Deviation Index"), use_container_width=True, hide_index=True)

    # ----------------------------------------------------
    # 4. ASYMMETRY POINTER & HIERARCHICAL ACCESS INDEX
    # ----------------------------------------------------
    st.header("⚖️ 4. Asymmetrical Vector Weight & Hierarchical Access Index")
    st.markdown("Analyzing spatial imbalance vectors to isolate reading direction pointers and hidden 'System Engineer' data channels.")
    
    if current_tags:
        left_side_nodes = 0
        right_side_nodes = 0
        left_angles_sum = 0.0
        right_angles_sum = 0.0
        
        for t in current_tags:
            coord_tuple = eval(t["Coordinate Location"])
            x_coord = coord_tuple[0]  # Isolated horizontal coordinate element
            raw_ang = float(t["Measured Axis Angle"].replace("°", ""))
            
            if x_coord < 128:
                left_side_nodes += 1
                left_angles_sum += raw_ang
            else:
                right_side_nodes += 1
                right_angles_sum += raw_ang
                
        total_nodes = left_side_nodes + right_side_nodes
        
        if total_nodes > 0:
            asymmetry_index = ((left_side_nodes - right_side_nodes) / total_nodes) * 100.0
            
            asym_col1, asym_col2 = st.columns(2)
            with asym_col1:
                st.metric(
                    label="🧭 Directional Asymmetry Index", 
                    value=f"{round(asymmetry_index, 1)}%",
                    delta="Left-Weighted Pointer" if asymmetry_index > 0 else "Right-Weighted Pointer" if asymmetry_index < 0 else "Perfect Balance (Zero Pointer)"
                )
                
                # Dynamic text feedback changes based on whether image is mirrored
                if view_mode == "Original Seal View (Carved Intaglio)":
                    if asymmetry_index > 15.0:
                        st.info("⬅️ **SEAL CARVING AXIS:** Asymmetry vectors confirm a native **Right-to-Left** engraving sweep.")
                    elif asymmetry_index < -15.0:
                        st.info("➡️ **SEAL CARVING AXIS:** Asymmetry vectors confirm an inverted **Left-to-Right** engraving sweep.")
                    else:
                        st.info("🔄 **SEAL CARVING AXIS:** Static geometric equilibrium.")
                else:
                    # Clay view logic: flips direction interpretations
                    if asymmetry_index > 15.0:
                        st.info("➡️ **CLAY READING FLOW:** Impression inversion projects an effective **Left-to-Right** reading flow.")
                    elif asymmetry_index < -15.0:
                        st.info("⬅️ **CLAY READING FLOW:** Impression inversion projects an effective **Right-to-Left** reading flow.")
                    else:
                        st.info("🔄 **CLAY READING FLOW:** Static geometric equilibrium.")
                    
            with asym_col2:
                unique_intervals = len(set([t["Harmonic Grid Link"] for t in current_tags]))
                
                if unique_intervals >= 4 and abs(asymmetry_index) > 10:
                    st.success("👑 **DATA ACCESS LEVEL: LEVEL 3 (SYSTEM ENGINEER)**  \nHidden data channel active. This sign contains highly asymmetric, multi-harmonic tags accessible only via specialized photometric angles.")
                elif unique_intervals >= 2:
                    st.warning("🤝 **DATA ACCESS LEVEL: LEVEL 2 (GUILD AUDITOR)**  \nStandard administrative verification layer. Tracks ledger metrics and commodity validation frameworks.")
                else:
                    st.error("🛒 **DATA ACCESS LEVEL: LEVEL 1 (PUBLIC MARKET)**  \nSurface level semantic token. Direct visual logograph representing standard commodity categorization.")
                    
            st.markdown("### 🛠️ Sub-Surface Geometric Weight Distribution")
            df_asym = pd.DataFrame([
                {"Matrix Quadrant": "Left Data Sector (< 128px Axis)", "Active Nodes": left_side_nodes, "Cumulative Angular Energy": f"{round(left_angles_sum, 1)}°"},
                {"Matrix Quadrant": "Right Data Sector (≥ 128px Axis)", "Active Nodes": right_side_nodes, "Cumulative Angular Energy": f"{round(right_angles_sum, 1)}°"}
            ])
            st.dataframe(df_asym, use_container_width=True, hide_index=True)
else:
    st.info("🖋️ Awaiting vector inputs. Draw an inscription loop above to trigger the 10-layer verification sweep sequence.")
