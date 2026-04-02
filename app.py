import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import time

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("🌀 IVC Symbolic Energy System — Coherence Engine")

# ------------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------------
st.sidebar.header("Controls")

num_symbols = st.sidebar.slider("Symbols", 3, 12, 6)
threshold = st.sidebar.slider("Cluster Threshold", 0.5, 0.95, 0.8)
run_animation = st.sidebar.toggle("▶️ Run Global Synchrony", False)
speed = st.sidebar.slider("Speed", 0.01, 0.2, 0.05)
field_mode = st.sidebar.toggle("🌊 Field Mode", True)

uploaded = st.sidebar.file_uploader("Upload Symbol Matrix (CSV)", type=["csv"])

# ------------------------------------------------------------------------------
# DATA INPUT
# ------------------------------------------------------------------------------
def generate_data(n):
    np.random.seed(42)
    symbols = [f"S{i}" for i in range(n)]
    values = np.random.rand(n, n)
    return pd.DataFrame(values, index=symbols, columns=symbols)

def load_data():
    if uploaded:
        df = pd.read_csv(uploaded, index_col=0)
        return df
    return generate_data(num_symbols)

# ------------------------------------------------------------------------------
# CORE COMPUTATION
# ------------------------------------------------------------------------------
def compute_resonance(df):
    data = df.to_numpy()
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / (norm + 1e-9)
    res = np.dot(normalized, normalized.T)
    return pd.DataFrame(res, index=df.index, columns=df.columns)

def compute_energy(matrix):
    return np.sum(matrix.values, axis=1)

def compute_flow(matrix):
    grad = np.gradient(matrix.values)
    flow = grad[0] + grad[1]
    v = np.mean(flow, axis=1)
    return np.stack([v, v, v], axis=1)

# ------------------------------------------------------------------------------
# COHERENCE ENGINE
# ------------------------------------------------------------------------------
def compute_coherence(matrix):
    M = matrix.values
    eigenvals = np.linalg.eigvals(M)
    coherence = np.max(np.real(eigenvals)) / (np.sum(np.abs(eigenvals)) + 1e-9)
    return float(coherence)

def coherence_label(ci):
    if ci > 0.6:
        return "🔥 HIGH COHERENCE (Aligned System)"
    elif ci > 0.3:
        return "⚡ EMERGING COHERENCE"
    else:
        return "🌑 LOW COHERENCE"

# ------------------------------------------------------------------------------
# CLUSTERING
# ------------------------------------------------------------------------------
def find_clusters(matrix, threshold):
    clusters = []
    visited = set()
    labels = matrix.index.tolist()

    for i in range(len(labels)):
        if i in visited:
            continue

        cluster = {labels[i]}
        for j in range(len(labels)):
            if i != j and matrix.iloc[i, j] >= threshold:
                cluster.add(labels[j])

        visited.update([labels.index(c) for c in cluster])
        clusters.append(cluster)

    return clusters

# ------------------------------------------------------------------------------
# EVOLUTION
# ------------------------------------------------------------------------------
def evolve(matrix, t=0.05):
    M = matrix.values
    phase = np.sin(np.linspace(0, 2*np.pi, M.shape[0]))[:, None]
    evolved = M + t * (np.dot(M, M.T) * phase)
    evolved = np.clip(evolved, 0, 1)
    return pd.DataFrame(evolved, index=matrix.index, columns=matrix.columns)

# ------------------------------------------------------------------------------
# VISUALS
# ------------------------------------------------------------------------------
def render_resonance(matrix, clusters):
    coords = PCA(n_components=3).fit_transform(matrix.values)
    labels = matrix.index.tolist()
    fig = go.Figure()

    for cluster in clusters:
        idx = [labels.index(s) for s in cluster]
        fig.add_trace(go.Scatter3d(
            x=coords[idx,0], y=coords[idx,1], z=coords[idx,2],
            mode="markers+text",
            text=[labels[i] for i in idx],
            marker=dict(size=6),
            name=str(cluster)
        ))
    fig.update_layout(title="3D Resonance Field", height=500)
    return fig

def render_flow(matrix, flow):
    coords = PCA(n_components=3).fit_transform(matrix.values)
    fig = go.Figure(data=go.Cone(
        x=coords[:,0], y=coords[:,1], z=coords[:,2],
        u=flow[:,0], v=flow[:,1], w=flow[:,2],
        sizeref=2
    ))
    fig.update_layout(title="Energy Flow Field", height=500)
    return fig

def render_frequency(energy, t):
    fig = go.Figure()
    freqs = np.linspace(0.1, 2.0, len(energy))

    for i, val in enumerate(energy):
        amp = val * (1 + 0.3*np.sin(t + freqs[i]))
        z = amp * np.sin(freqs[i]*np.pi + t)

        fig.add_trace(go.Scatter3d(
            x=[freqs[i], freqs[i]],
            y=[0, amp],
            z=[0, z],
            mode="lines"
        ))

    fig.update_layout(title="Frequency Spectrum", height=500)
    return fig

def render_field(matrix, t):
    coords = PCA(n_components=2).fit_transform(matrix.values)

    x = np.linspace(-2,2,50)
    y = np.linspace(-2,2,50)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)

    for i,(sx,sy) in enumerate(coords):
        energy = np.sum(matrix.iloc[i])
        dist = np.sqrt((X-sx)**2 + (Y-sy)**2)
        Z += energy * np.sin(5*dist - t)

    Z /= np.max(np.abs(Z)) + 1e-9

    fig = go.Figure(data=go.Surface(x=X,y=Y,z=Z))
    fig.update_layout(title="🌊 Energy Field", height=600)
    return fig

# ------------------------------------------------------------------------------
# INIT
# ------------------------------------------------------------------------------
if "matrix" not in st.session_state:
    data = load_data()
    st.session_state.matrix = compute_resonance(data)

# ------------------------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
field_area = st.container()

# ------------------------------------------------------------------------------
# RENDER
# ------------------------------------------------------------------------------
def render_all(matrix, t):
    clusters = find_clusters(matrix, threshold)
    energy = compute_energy(matrix)
    flow = compute_flow(matrix)

    ci = compute_coherence(matrix)

    col1.metric("Coherence Index", f"{ci:.3f}")
    col2.metric("State", coherence_label(ci))

    col1.plotly_chart(render_resonance(matrix, clusters), use_container_width=True, key=f"res_{t}")
    col2.plotly_chart(render_flow(matrix, flow), use_container_width=True, key=f"flow_{t}")
    col3.plotly_chart(render_frequency(energy, t), use_container_width=True, key=f"freq_{t}")

    if field_mode:
        field_area.plotly_chart(render_field(matrix, t), use_container_width=True, key=f"field_{t}")

# ------------------------------------------------------------------------------
# LOOP
# ------------------------------------------------------------------------------
if run_animation:
    for step in range(60):
        st.session_state.matrix = evolve(st.session_state.matrix)
        render_all(st.session_state.matrix, step)
        time.sleep(speed)
else:
    render_all(st.session_state.matrix, 0)
