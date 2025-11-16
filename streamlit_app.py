# streamlit_app.py

import streamlit as st
import numpy as np
from simulator.simulator import load_signs_json, create_grid

st.title("Indus Script Holographic-Frequency Simulator")

# -----------------------------
# Load database
# -----------------------------
try:
    signs = load_signs_json("data/nb_signs.json")
except Exception as e:
    st.error(f"Error loading signs.json: {e}")
    st.stop()

# -----------------------------
# Sidebar selection
# -----------------------------
nb_list = sorted(signs.keys())
selected_nb = st.sidebar.selectbox("Select Sign (NB)", nb_list)

mode = st.sidebar.selectbox(
    "Holographic Mode",
    ["acoustic", "light", "matrix"]
)

symbol = signs[selected_nb]

# -----------------------------
# Display metadata
# -----------------------------
st.subheader(f"Metadata for {selected_nb}")

st.json({
    "icit_id": symbol.icit_id,
    "class": symbol.sign_class,
    "set": symbol.set,
    "frequency": symbol.frequency,
    "positions": symbol.positions,
    "sites": symbol.sites,
})

# -----------------------------
# Generate holographic matrix
# -----------------------------
matrix = symbol.generate_wave_matrix(mode=mode)
grid = create_grid(matrix)

# -----------------------------
# Display visualization
# -----------------------------
st.subheader("Holographic Waveform")
st.image(
    grid,
    width=400,
    caption=f"{mode.title()} Mode Waveform for {selected_nb}"
)
