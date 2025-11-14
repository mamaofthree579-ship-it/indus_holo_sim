import streamlit as st
from PIL import Image

# Core simulator imports
from simulator.symbol import Symbol
from simulator.simulator import HoloSimulator, create_grid
from simulator.indus_signs import INDUS_SIGNS, NB_LIST

# NEW REQUIRED IMPORT
from simulator.glyph_generator import generate_glyph


st.set_page_config(page_title="Indus Holographic Simulator", layout="wide")

st.title("Indus Script — Holographic Frequency Simulator")


# ---------------------------------------------------
# Sidebar: Choose glyph rendering mode
# ---------------------------------------------------
mode = st.sidebar.selectbox(
    "Glyph Rendering Mode",
    ["neutral", "acoustic", "light", "matrix"],
    index=1
)

# ---------------------------------------------------
# Sidebar: Choose NB Sign
# ---------------------------------------------------
selected_nb = st.sidebar.selectbox("NB Code", NB_LIST)

entry = INDUS_SIGNS[selected_nb]

glyph_path = entry.get("image_url")

if not glyph_path:
    glyph_path = generate_glyph(selected_nb, mode=mode)
    entry["image_url"] = glyph_path

if glyph_path:
    try:
        img = Image.open(glyph_path)
        st.sidebar.image(img, caption=f"{selected_nb} — {entry['name']}", use_column_width=True)
    except:
        st.sidebar.write("(glyph image could not be loaded)")


# ---------------------------------------------------
# Symbol parameters
# ---------------------------------------------------
freq = st.sidebar.number_input("Base Frequency", 10.0, 2000.0, 440.0)
strength = st.sidebar.number_input("Strength", 0.1, 10.0, 1.0)

# Harmonics list
harmonics = []
num_harm = st.sidebar.slider("Number of Harmonics", 1, 6, 3)

st.sidebar.write("Harmonic multipliers:")

for i in range(num_harm):
    multiplier = st.sidebar.number_input(
        f"Multiplier {i+1}",
        min_value=1.0,
        max_value=10.0,
        value=float(i+1)
    )
    harmonics.append(multiplier)


# ---------------------------------------------------
# Add symbol to simulation
# ---------------------------------------------------
if "symbols" not in st.session_state:
    st.session_state.symbols = []

if st.sidebar.button("Add Symbol"):
    symbol = Symbol(
        name=selected_nb,
        base_freq=freq,
        harmonics=harmonics,
        strength=strength,
        image_path=glyph_path
    )
    st.session_state.symbols.append(symbol)


# ---------------------------------------------------
# Run simulation
# ---------------------------------------------------
st.header("Simulation")

grid_size = st.slider("Grid Size", 100, 400, 200)
x, y = create_grid(grid_size)

sim = HoloSimulator()

if st.session_state.symbols:
    field = sim.compute_field(x, y, st.session_state.symbols)

    st.subheader("Combined Field Amplitude")

    # Use Streamlit's built-in image display
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(np.abs(field), cmap="viridis")
    ax.set_title("Holographic Interference")
    ax.axis("off")

    st.pyplot(fig)

else:
    st.write("Add at least one symbol to run the simulation.")
