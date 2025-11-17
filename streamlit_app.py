import streamlit as st
from simulator.simulator import load_signs_json, create_grid, HoloSimulator, Symbol
from pathlib import Path
import os

st.title("Indus Holographic Simulator — Glyph Stimuli")

# -----------------------------
# Load database
# -----------------------------

signs = load_signs_json("data/nb_signs.json")
nb = st.sidebar.selectbox("Select NB sign", sorted(signs.keys()))

# Choose whether to use synthetic or glyph-driven
mode = st.sidebar.radio("Stimulus", ["synthetic", "glyph"])
if mode == "glyph":
    glyph_dir = Path("data/images")
    glyph_path = glyph_dir / f"{nb}_mask.png"
    if not glyph_path.exists():
        st.warning(f"No glyph mask found for {nb}. Run glyph generator to create {glyph_path}.")
    else:
        stim_mode = st.sidebar.selectbox("Field type", ["holo","light","acoustic","mask"])
        symbol = signs[nb]
        holo = HoloSimulator()
        fig = holo.compute_field_from_glyph(symbol, str(glyph_path), mode=stim_mode)
        st.pyplot(fig)
else:
    # synthetic
    symbol = signs[nb]
    holo = HoloSimulator()
    grid = create_grid(256)
    fig = holo.compute_field(symbol, grid)
    st.pyplot(fig)


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
