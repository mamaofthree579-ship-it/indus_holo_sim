import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Generate Sample Data
# ------------------------------------------------------------------------------
def generate_sample_data(n=5):
    """Generate symbolic test data."""
    np.random.seed(42)
    symbols = [f"Symbol_{i}" for i in range(n)]
    values = np.random.rand(n, n)
    return pd.DataFrame(values, index=symbols, columns=symbols)


# ------------------------------------------------------------------------------
# Resonance Matrix
# ------------------------------------------------------------------------------
def compute_resonance_matrix(df):
    """Compute cosine similarity matrix."""
    data = df.to_numpy()
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / (norm + 1e-9)
    resonance = np.dot(normalized, normalized.T)
    return pd.DataFrame(resonance, index=df.index, columns=df.columns)


# ------------------------------------------------------------------------------
# Cluster Detection
# ------------------------------------------------------------------------------
def find_resonant_clusters(matrix, threshold=0.8):
    """Find clusters of strongly resonant symbols."""
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
# Symbol Energy
# ------------------------------------------------------------------------------
def compute_symbol_energy(matrix):
    """Return energy per symbol as dictionary."""
    energy = matrix.sum(axis=1)
    return {symbol: float(val) for symbol, val in energy.items()}


# ------------------------------------------------------------------------------
# Energy Flow (Vector Field)
# ------------------------------------------------------------------------------
def compute_energy_flow(matrix):
    """Compute simple vector field from matrix gradients."""
    data = matrix.to_numpy()

    grad_x = np.gradient(data, axis=0)
    grad_y = np.gradient(data, axis=1)

    flow = grad_x + grad_y

    # Reduce to one vector per symbol
    vectors = np.mean(flow, axis=1)

    # Expand to 3D vectors
    vectors_3d = np.stack([vectors, vectors, vectors], axis=1)

    return vectors_3d


# ------------------------------------------------------------------------------
# Evolution Step (for animation)
# ------------------------------------------------------------------------------
def evolve_matrix_step(matrix, t=0.05):
    """Evolve matrix over time (oscillatory system)."""
    M = matrix.to_numpy()

    phase = np.sin(np.linspace(0, 2 * np.pi, M.shape[0]))[:, None]
    evolved = M + t * (np.dot(M, M.T) * phase)

    evolved = np.clip(evolved, 0, 1)

    return pd.DataFrame(evolved, index=matrix.index, columns=matrix.columns)
