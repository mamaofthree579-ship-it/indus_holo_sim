import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from data.nb_signs import Symbol
from simulator.simulator import HoloSimulator
from simulator.grid import create_grid


# --------------------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Indus Holographic-Frequency Simulator",
    layout="wide"
)

st.title("🌀 Indus Script Holographic–Frequency Simulator")
st.write("Explore wave-based holographic interference patterns encoded by symbolic nodes.")


# --------------------------------------------------------------------------------------
# Grid setup
# --------------------------------------------------------------------------------------
GRID_SIZE = 256
XX, YY = create_grid(GRID_SIZE)

if "symbols" not in st.session_state:
    st.session_state.symbols = []


# --------------------------------------------------------------------------------------
# Sidebar: Add symbol
# --------------------------------------------------------------------------------------
st.sidebar.header("➕ Add a Symbol")

with st.sidebar.form("add_symbol"):
    name = st.text_input("Symbol Name", "symbol")
    x = st.slider("X Position", 0.0, 1.0, 0.5)
    y = st.slider("Y Position", 0.0, 1.0, 0.5)

    base_freq = st.number_input("Base Frequency (Hz)", 1.0, 200.0, 20.0)
    amplitude = st.number_input("Amplitude", 0.1, 5.0, 1.0)
    sigma = st.number_input("Spatial Sigma", 0.01, 0.2, 0.06)

    st.write("---")
    st.write("### Harmonics")
    num_harm = st.number_input("Number of Harmonics", 1, 6, 1)

    harmonics = []
    for i in range(num_harm):
        st.write(f"**Harmonic {i+1}**")
        mult = st.number_input(f"Multiplier {i+1}", 1.0, 10.0, float(i+1))
        rel_amp = st.number_input(f"Relative Amp {i+1}", 0.0, 2.0, 1.0)
        phase = st.number_input(f"Phase Offset {i+1}", 0.0, np.pi, 0.0)
        harmonics.append((mult, rel_amp, phase))

    submitted = st.form_submit_button("Add Symbol")

if submitted:
    s = Symbol(
        name=name,
        x=x,
        y=y,
        base_freq=base_freq,
        amplitude=amplitude,
        sigma=sigma,
        harmonics=harmonics
    )
    st.session_state.symbols.append(s)
    st.success(f"Added symbol '{name}'.")


# --------------------------------------------------------------------------------------
# Sidebar: Symbol list + clear
# --------------------------------------------------------------------------------------
st.sidebar.write("---")
st.sidebar.header("📜 Symbols")

if st.session_state.symbols:
    for s in st.session_state.symbols:
        st.sidebar.write(f"• **{s.name}** — {s.base_freq} Hz")
else:
    st.sidebar.write("No symbols added yet.")

if st.sidebar.button("Clear All Symbols"):
    st.session_state.symbols = []
    st.sidebar.success("Cleared.")


# --------------------------------------------------------------------------------------
# Simulation settings
# --------------------------------------------------------------------------------------
st.sidebar.write("---")
st.sidebar.header("⚙ Simulation Settings")

time_samples = st.sidebar.slider("Time Samples", 10, 200, 60)
t_max = st.sidebar.slider("Time Window (seconds)", 0.1, 5.0, 1.0)

run_button = st.sidebar.button("Run Simulation")


# --------------------------------------------------------------------------------------
# Run simulation
# --------------------------------------------------------------------------------------
if run_button:
    if not st.session_state.symbols:
        st.warning("Add at least one symbol first.")
    else:
        sim = HoloSimulator(XX, YY, time_samples, t_max)

        for s in st.session_state.symbols:
            sim.add_symbol(s)

        with st.spinner("Computing holographic interference…"):
            intensity = sim.simulate()

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(intensity, origin="lower", extent=(0, 1, 0, 1))
        ax.set_title("Holographic Intensity Field")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        for s in st.session_state.symbols:
            ax.plot(s.x, s.y, "wo")
            ax.text(s.x + 0.01, s.y + 0.01, s.name, color="white", fontsize=8)

        st.pyplot(fig)

else:
    st.info("Add symbols and click **Run Simulation** to generate holographic patterns.")


st.write("---")
st.caption("Indus Holographic-Frequency Research Simulator – built with Streamlit.")
