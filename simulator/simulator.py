# simulator/simulator.py
"""
Grid generator and HoloSimulator.

- create_grid(size) -> returns (XX, YY) as 2D numpy arrays in normalized [0..1] coords
- HoloSimulator.compute_field(symbol, grid) -> matplotlib.figure.Figure
    It computes a simple interference intensity map (time-instant snapshot)
    and returns a Matplotlib figure for direct st.pyplot(...) display.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple

def create_grid(size: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create normalized 0..1 grid of shape (size,size).
    Returns (XX, YY).
    """
    xs = np.linspace(0.0, 1.0, size)
    ys = np.linspace(0.0, 1.0, size)
    XX, YY = np.meshgrid(xs, ys)
    return XX, YY


class HoloSimulator:
    def __init__(self):
        pass

    def compute_field(self, symbol, grid, time: float = 0.0) -> Figure:
        """
        Compute a holographic intensity snapshot for a single symbol (or object with same API).

        Args:
            symbol: Symbol object (has frequency, harmonics, amplitude, x, y, sigma)
            grid: tuple (XX, YY) as returned by create_grid
            time: optional time parameter (unused for now except phase offsets)

        Returns:
            Matplotlib Figure object containing plotted intensity map.
        """
        XX, YY = grid
        # distances from symbol location (normalized coords)
        dx = XX - float(symbol.x)
        dy = YY - float(symbol.y)
        r = np.sqrt(dx**2 + dy**2) + 1e-9  # avoid zero

        # spatial envelope (Gaussian)
        sigma = float(symbol.sigma)
        env = np.exp(-0.5 * (r**2) / (sigma**2))

        # base angular term (use 2*pi*freq as spatial angular factor)
        # (This is a simple toy model for interference; adjust scaling as you like.)
        k0 = 2.0 * np.pi * float(symbol.frequency)

        # complex field initialised to zero
        field = np.zeros_like(XX, dtype=np.complex128)

        # base harmonic
        field += symbol.amplitude * env * np.exp(1j * (k0 * (-r) + 2.0 * np.pi * symbol.frequency * time))

        # additional harmonics
        for (mult, rel, ph) in symbol.harmonics:
            mult = float(mult)
            rel = float(rel)
            ph = float(ph)
            k = 2.0 * np.pi * (symbol.frequency * mult)
            field += symbol.amplitude * rel * env * np.exp(1j * (k * (-r) + ph + 2.0 * np.pi * (symbol.frequency * mult) * time))

        # intensity (magnitude squared)
        intensity = np.abs(field) ** 2
        # normalize for display
        intensity = intensity / (intensity.max() + 1e-12)

        # create matplotlib figure to return
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(intensity, origin="lower", cmap="magma", extent=(0, 1, 0, 1))
        ax.set_title(f"Holographic intensity — {symbol.name}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized intensity")

        return fig
