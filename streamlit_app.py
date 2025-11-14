# streamlit_app.py

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Imports from your package
from simulator.symbol import Symbol
from simulator.simulator import HoloSimulator, create_grid
from simulator.indus_signs import INDUS_SIGNS, NB_LIST
from simulator.glyph_generator import generate_glyph

st.set_page_config(page_title="Indus Holographic Simulator", layout="wide")
st.title("Indus Script — Holographic Frequency Simulator")

# ---------------------------------------------------
# Sidebar: Glyph rendering mode and NB selector
# ---------------------------------------------------
mode = st.sidebar.selectbox("Glyph Rendering Mode", ["neutral", "acoustic", "light", "matrix"], index=1)
selected_nb = st.sidebar.selectbox("NB Code", NB_LIST)
entry = INDUS_SIGNS[selected_nb]

glyph_path = entry.get("image_url")
if not glyph_path:
    glyph_path = generate_glyph(selected_nb, mode=mode)
    entry["image_url"] = glyph_path

if glyph_path:
    try:
        im = Image.open(glyph_path)
        st.sidebar.image(im, caption=f"{selected_nb} — {entry['name']}", use_column_width=True)
    except Exception as e:
        st.sidebar.write("(glyph image could not be loaded)")

# ---------------------------------------------------
# Symbol parameter inputs
# ---------------------------------------------------
st.sidebar.write("### Symbol parameters for new symbol")
freq = st.sidebar.number_input("Base Frequency (Hz)", min_value=1.0, max_value=2000.0, value=20.0, step=1.0)
amplitude = st.sidebar.number_input("Amplitude", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
xpos = st.sidebar.slider("X position", 0.0, 1.0, 0.5)
ypos = st.sidebar.slider("Y position", 0.0, 1.0, 0.5)

# harmonic controls (ensure float defaults)
num_harm = st.sidebar.slider("Number of harmonics", 1, 6, 1)
harmonics = []
for i in range(num_harm):
    mult = float(st.sidebar.number_input(f"Multiplier {i+1}", min_value=1.0, max_value=10.0, value=float(i+1)))
    rel = float(st.sidebar.number_input(f"Rel amp {i+1}", min_value=0.0, max_value=2.0, value=1.0))
    phase = float(st.sidebar.number_input(f"Phase {i+1}", min_value=0.0, max_value=3.1416, value=0.0))
    harmonics.append((mult, rel, phase))

# ---------------------------------------------------
# Symbol list state and add button
# ---------------------------------------------------
if "symbols" not in st.session_state:
    st.session_state.symbols = []

st.sidebar.write("---")
if st.sidebar.button("Add Symbol"):
    s = Symbol(
        name=selected_nb,
        base_freq=freq,
        harmonics=harmonics,
        image_path=glyph_path,
        x=xpos,
        y=ypos,
        amplitude=amplitude
    )
    st.session_state.symbols.append(s)
    st.sidebar.success(f"Added {selected_nb}")

if st.sidebar.button("Clear symbols"):
    st.session_state.symbols = []
    st.sidebar.success("Cleared symbols")

# show current symbols
st.sidebar.write("### Current symbols")
for s in st.session_state.symbols:
    st.sidebar.write(f"- {s.name} @ ({s.x:.2f},{s.y:.2f}) f={s.base_freq}Hz amp={s.amplitude}")

# ---------------------------------------------------
# Grid / simulation controls
# ---------------------------------------------------
grid_size = st.slider("Grid Size (pixels)", 100, 400, 200)
xgrid, ygrid = create_grid(grid_size)

sim = HoloSimulator()

st.header("Simulation")
if st.session_state.symbols:
    field = sim.compute_field(xgrid, ygrid, st.session_state.symbols)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(np.abs(field), cmap="magma", origin="lower", extent=(0,1,0,1))
    ax.set_title("Holographic Interference (magnitude)")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("Add at least one symbol using the sidebar to run the simulation.")
