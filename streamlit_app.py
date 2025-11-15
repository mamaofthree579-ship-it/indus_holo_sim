# streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
import io
import time
from pathlib import Path

from simulator.symbol import Symbol
from simulator.simulator import HoloSimulator, create_grid
from simulator.indus_signs import INDUS_SIGNS, NB_LIST
from simulator.glyph_generator import generate_glyph

st.set_page_config(page_title="Indus Holographic Simulator", layout="wide")
st.title("🌀 Indus Script — Holographic Frequency Simulator")

# ---------------- Symbol parameter inputs ----------------
st.sidebar.write("## New Symbol Parameters")
freq = float(st.sidebar.number_input("Base frequency (Hz)", min_value=1.0, max_value=2000.0, value=entry.get("default_freq", 20.0)))
amplitude = float(st.sidebar.number_input("Amplitude", min_value=0.01, max_value=10.0, value=1.0))
xpos = float(st.sidebar.slider("X position", 0.0, 1.0, 0.5))
ypos = float(st.sidebar.slider("Y position", 0.0, 1.0, 0.5))
sigma = float(st.sidebar.number_input("Spatial sigma", min_value=0.01, max_value=0.5, value=entry.get("sigma", 0.06)))

num_harm = st.sidebar.slider("Harmonics", 1, 6, max(1, len(entry.get("harmonics", []))))
harmonics = []
st.sidebar.write("Harmonic params (mult, rel, phase)")
# try to use defaults from entry
default_h = entry.get("harmonics", [[1.0,1.0,0.0]])
for i in range(num_harm):
    d = default_h[i] if i < len(default_h) else [i+1,1.0,0.0]
    mult = float(st.sidebar.number_input(f"Multiplier {i+1}", min_value=1.0, max_value=10.0, value=float(d[0])))
    rel = float(st.sidebar.number_input(f"Rel amp {i+1}", min_value=0.0, max_value=5.0, value=float(d[1])))
    phase = float(st.sidebar.number_input(f"Phase {i+1}", min_value=0.0, max_value=6.28318, value=float(d[2])))
    harmonics.append((mult, rel, phase))

# ---------------- Symbol list state ----------------
if "symbols" not in st.session_state:
    st.session_state.symbols = []

st.sidebar.write("---")
if st.sidebar.button("Add symbol to scene"):
    s = Symbol(
        name=selected_nb,
        base_freq=freq,
        harmonics=harmonics,
        image_path=glyph_path,
        amplitude=amplitude,
        x=xpos,
        y=ypos,
        sigma=sigma
    )
    st.session_state.symbols.append(s)
    st.sidebar.success(f"Added {selected_nb}")

if st.sidebar.button("Clear symbols"):
    st.session_state.symbols = []
    st.sidebar.success("Cleared symbols")

st.sidebar.write("### Current symbols")
for i,s in enumerate(st.session_state.symbols):
    st.sidebar.write(f"{i+1}. {s.name} f={s.base_freq}Hz pos=({s.x:.2f},{s.y:.2f}) amp={s.amplitude}")

# ---------------- Simulation controls ----------------
st.write("## Simulation Controls")
col1, col2 = st.columns([1,2])

with col1:
    grid_size = st.slider("Grid size (px)", 100, 400, 200)
    time_samples = st.slider("Time samples (for animation)", 4, 120, 24)
    t_max = st.slider("Time window (s)", 0.1, 2.0, 0.8)
    animate = st.checkbox("Render animation (time evolution)", value=False)

with col2:
    st.write("Use the canvas below to preview interference.")

# ---------------- Compute & render ----------------
XX, YY = create_grid(grid_size)
sim = HoloSimulator()

if not st.session_state.symbols:
    st.info("No symbols yet — add symbols from the sidebar to begin.")
else:
    if not animate:
        # single time snapshot at t=0
        field = sim.compute_field(XX, YY, st.session_state.symbols, times=0.0)
        intensity = np.abs(field)
        intensity /= intensity.max() + 1e-12

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(intensity, origin="lower", extent=(0,1,0,1), cmap="inferno")
        ax.set_title("Holographic Intensity (snapshot)")
        ax.axis("off")
        st.pyplot(fig)
    else:
        # generate frames and show animation; provide download
        times = np.linspace(0.0, t_max, time_samples)
        frames = []
        progress = st.progress(0)
        for i,t in enumerate(times):
            field_t = sim.compute_field(XX, YY, st.session_state.symbols, times=float(t))
            mag = np.abs(field_t)
            mag /= mag.max() + 1e-12
            # convert to 8-bit image
            img = (255 * plt.cm.inferno(mag)[:, :, :3]).astype(np.uint8)
            frames.append(img)
            progress.progress((i+1)/len(times))
        progress.empty()

        # display animation inline as GIF
        buf = io.BytesIO()
        imageio.mimsave(buf, frames, format="GIF", fps=max(4, int(len(times)/t_max)))
        st.image(buf.getvalue(), format="GIF")
        st.download_button("Download GIF", data=buf.getvalue(), file_name="hologram.gif", mime="image/gif")

st.write("---")
st.caption("Tip: generate glyphs once (left sidebar) — they are cached in data/images/ for reuse.")
