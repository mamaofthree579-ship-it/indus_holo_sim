import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from simulator import Symbol, HoloSimulator, create_grid

# --------------------------------------------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Indus Holographic-Frequency Simulator",
    layout="wide"
)

st.title("🌀 Indus Script Holographic–Frequency Simulator")
st.write("A research tool for exploring wave-based interference patterns encoded by symbolic nodes.")

# --------------------------------------------------------------------------------------
# Grid & Simulator Setup
# --------------------------------------------------------------------------------------
GRID_SIZE = 256
XX, YY = create_grid(GRID_SIZE)

# State storage
if "symbols" not in st.session_state:
    st.session_state.symbols = []

# --------------------------------------------------------------------------------------
# Sidebar – Symbol Controls
# --------------------------------------------------------------------------------------
st.sidebar.header("➕ Add a Symbol")

with st.sidebar.form("new_symbol_form"):
    name = st.text_input("Name", "symbol")
    x = st.slider("X Position", 0.0, 1.0, 0.5)
    y = st.slider("Y Position", 0.0, 1.0, 0.5)

    base_freq = st.number_input("Base Frequency (Hz)", 1.0, 200.0, 20.0)
    amplitude = st.number_input("Amplitude", 0.1, 5.0, 1.0)
    sigma = st.number_input("Spatial Spread (sigma)", 0.01, 0.2, 0.06)

    # Harmonic parameters
    st.write("---")
    st.write("### Harmonics")
    num_harm = st.number_input("Number of Harmonics", 1, 6, 1)

    harmonic_params = []
    for i in range(num_harm):
        st.write(f"**Harmonic {i+1}**")
        mult = st.number_input(f"Multiplier {i+1}", 1.0, 10.0, float(i+1))
        rel_amp = st.number_input(f"Relative Amplitude {i+1}", 0.0, 2.0, 1.0)
        phase = st.number_input(f"Phase Offset {i+1}", 0.0, 3.14, 0.0)
        harmonic_params.append((mult, rel_amp, phase))

    submitted = st.form_submit_button("Add Symbol")

if submitted:
    new_symbol = Symbol(
        name=name,
        x=x,
        y=y,
        base_freq=base_freq,
        amplitude=amplitude,
        sigma=sigma,
        harmonics=harmonic_params,
    )
    st.session_state.symbols.append(new_symbol)
    st.success(f"Added symbol '{name}' to the simulation.")


# --------------------------------------------------------------------------------------
# Symbol List & Clear Button
# --------------------------------------------------------------------------------------
st.sidebar.write("---")
st.sidebar.header("📜 Current Symbols")

if st.session_state.symbols:
    for s in st.session_state.symbols:
        st.sidebar.write(f"• **{s.name}** — freq {s.base_freq} Hz")
else:
    st.sidebar.write("No symbols yet.")

if st.sidebar.button("Clear All Symbols"):
    st.session_state.symbols = []
    st.sidebar.success("All symbols cleared.")

# --------------------------------------------------------------------------------------
# Simulation Controls
# --------------------------------------------------------------------------------------
st.sidebar.write("---")
st.sidebar.header("⚙ Simulation Settings")

time_samples = st.sidebar.slider("Time Samples", 10, 200, 60)
t_max = st.sidebar.slider("Time Window (seconds)", 0.1, 5.0, 1.0)

run_sim = st.sidebar.button("Run Simulation")


# --------------------------------------------------------------------------------------
# Run Simulation & Plot
# --------------------------------------------------------------------------------------
if run_sim:
    if not st.session_state.symbols:
        st.warning("Add at least one symbol before simulating.")
    else:
        sim = HoloSimulator(XX, YY, time_samples=time_samples, t_max=t_max)

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
            ax.plot(s.x, s.y, "wo", markersize=5)
            ax.text(s.x + 0.01, s.y + 0.01, s.name, color="white", fontsize=8)

        st.pyplot(fig)

else:
    st.info("Add symbols and click **Run Simulation** to generate the holographic pattern.")

# --------------------------------------------------------------------------------------
st.write("---")
st.caption("Indus Holographic-Frequency Research Environment – built with Streamlit.")
