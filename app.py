import streamlit as st
import numpy as np
import time
import sys
import os
import src.vector_plot as vp

st.write("Loaded:", dir(vp))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Clear cache to force updates
st.cache_data.clear()

# Imports
from modules.analytics import (
    generate_sample_data,
    compute_resonance_matrix,
    find_resonant_clusters,
    compute_symbol_energy,
    compute_energy_flow,
    evolve_matrix_step
)

from src.vector_plot import (
    render_3d_resonance_field,
    render_energy_flow_field,
    render_frequency_spectrum
)

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("🌀 IVC Symbolic Energy System (Global Synchrony Mode)")

# ------------------------------------------------------------------------------
# CONTROLS
# ------------------------------------------------------------------------------
st.sidebar.header("Controls")

num_symbols = st.sidebar.slider("Symbols", 3, 15, 6)
threshold = st.sidebar.slider("Cluster Threshold", 0.5, 0.95, 0.8)

run_animation = st.sidebar.toggle("▶️ Run Global Synchrony", value=False)
speed = st.sidebar.slider("Speed", 0.01, 0.2, 0.05)

# ------------------------------------------------------------------------------
# INITIAL DATA
# ------------------------------------------------------------------------------
if "matrix" not in st.session_state:
    data = generate_sample_data(num_symbols)
    matrix = compute_resonance_matrix(data)
    st.session_state.matrix = matrix

# ------------------------------------------------------------------------------
# PLACEHOLDERS (important for live updates)
# ------------------------------------------------------------------------------
col1, col2 = st.columns(2)
col3 = st.container()

res_placeholder = col1.empty()
flow_placeholder = col2.empty()
freq_placeholder = col3.empty()

# ------------------------------------------------------------------------------
# GLOBAL SYNCHRONY LOOP
# ------------------------------------------------------------------------------
def render_all(matrix):
    clusters = find_resonant_clusters(matrix, threshold=threshold)
    energy_map = compute_symbol_energy(matrix)
    flow_vectors = compute_energy_flow(matrix)

    fig_res = render_3d_resonance_field(matrix, clusters)
    fig_flow = render_energy_flow_field(matrix, flow_vectors)
    fig_freq = render_frequency_spectrum(energy_map)

    res_placeholder.plotly_chart(fig_res, use_container_width=True, key="res")
    flow_placeholder.plotly_chart(fig_flow, use_container_width=True, key="flow")
    freq_placeholder.plotly_chart(fig_freq, use_container_width=True, key="freq")


# ------------------------------------------------------------------------------
# RUN MODE
# ------------------------------------------------------------------------------
if run_animation:
    for _ in range(50):  # number of cycles
        st.session_state.matrix = evolve_matrix_step(st.session_state.matrix)

        render_all(st.session_state.matrix)

        time.sleep(speed)

    st.sidebar.success("Cycle complete")
else:
    render_all(st.session_state.matrix)
