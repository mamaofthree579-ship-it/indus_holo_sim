import streamlit as st
import numpy as np
from simulator.simulator import load_signs_json, create_grid

st.title("Indus Script Holographic-Frequency Simulator")

# Load database
signs = load_signs_json("nb_signs.json")

# Sidebar
nb_list = sorted(signs.keys())
selected_nb = st.sidebar.selectbox("Select Sign (NB)", nb_list)

mode = st.sidebar.selectbox(
    "Holographic Mode",
    ["acoustic", "light", "matrix"]
)

symbol = signs[selected_nb]

# Display metadata
st.subheader(f"Metadata for {selected_nb}")

st.json({
    "icit_id": symbol.icit_id,
    "class": symbol.sign_class,
    "set": symbol.set,
    "frequency": symbol.frequency,
    "positions": symbol.positions,
    "sites": symbol.sites
})

# Generate matrix
matrix = symbol.generate_wave_matrix(mode=mode)
grid = create_grid(matrix)

# Visualization
st.subheader("Holographic Waveform")
st.image(grid, width=400, caption=f"{mode.title()} Mode")
