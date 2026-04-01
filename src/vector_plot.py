# ------------------------------------------------------------------------------
# FIELD MODE — Continuous Energy Surface
# ------------------------------------------------------------------------------
def render_energy_field(matrix, resolution=50, time_phase=0.0):
    """
    Render continuous energy field using wave interference.
    Each symbol acts like a wave source.
    """

    import numpy as np
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    if matrix is None or matrix.empty:
        raise ValueError("Matrix is empty")

    # Reduce symbols to 2D positions
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix.values)

    # Create grid
    x = np.linspace(-2, 2, resolution)
    y = np.linspace(-2, 2, resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)

    # Build wave field from symbols
    for i, (sx, sy) in enumerate(coords):
        energy = np.sum(matrix.iloc[i].values)

        # Distance field
        dist = np.sqrt((X - sx)**2 + (Y - sy)**2)

        # Wave contribution
        wave = energy * np.sin(5 * dist - time_phase)

        Z += wave

    # Normalize
    Z = Z / (np.max(np.abs(Z)) + 1e-9)

    # Create surface
    fig = go.Figure(data=[
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Viridis"
        )
    ])

    fig.update_layout(
        title="🌊 Symbolic Energy Field (Wave Interference)",
        scene=dict(
            xaxis_title="Field X",
            yaxis_title="Field Y",
            zaxis_title="Energy Amplitude"
        ),
        height=700
    )

    return fig
