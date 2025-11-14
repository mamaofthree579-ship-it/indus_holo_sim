import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from simulator.symbol import Symbol
from simulator.simulator import HoloSimulator
from simulator.grid import create_grid
from simulator.indus_signs import INDUS_SIGNS, NB_LIST


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
    
    st.write("### Choose Indus Sign")

    nb_code = st.selectbox("NB Sign ID", NB_LIST)
    sign_info = INDUS_SIGNS[nb_code]

    name = st.text_input("Custom Name (optional)", sign_info["name"])

    x = st.slider("X Position", 0.0, 1.0, 0.5)
    y = st.slider("Y Position", 0.0, 1.0, 0.5)

    base_freq = st.number_input(
        "Base Frequency (Hz)",
        1.0, 200.0,
        sign_info["default_freq"]
    )

    amplitude = st.number_input("Amplitude", 0.1, 5.0, 1.0)

    sigma = st.number_input(
        "Spatial Sigma",
        0.01, 0.2,
        sign_info["sigma"]
    )

    st.write("### Harmonics")
    
    default_harm = sign_info["harmonics"]
    harmonic_count = len(default_harm)

    harmonics = []
    for i in range(harmonic_count):
        mult, rel_amp, phase = default_harm[i]
        st.write(f"**Harmonic {i+1}**")
        mult = st.number_input(f"Multiplier {i+1}", float(1.0), float(10.0), float(mult))
        rel_amp = st.number_input(f"Relative Amp {i+1}", float(0.0), float(2.0), float(rel_amp))
        phase = st.number_input(f"Phase Offset {i+1}", float(0.0), float(np.pi), float(phase))
        harmonics.append((mult, rel_amp, phase))

    submitted = st.form_submit_button("Add Sign")

if submitted:
    s = Symbol(
        name=name if name else sign_info["name"],
        x=x,
        y=y,
        base_freq=base_freq,
        amplitude=amplitude,
        sigma=sigma,
        harmonics=harmonics
    )

    st.session_state.symbols.append(s)
    st.success(f"Added Indus Sign {nb_code} — {sign_info['name']}.")


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
