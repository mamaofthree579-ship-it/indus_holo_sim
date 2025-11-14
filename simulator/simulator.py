# simulator/simulator.py

import numpy as np
from simulator.symbol import Symbol


def create_grid(size: int):
    """
    Create a centered 2D grid scaled -1..1 in both axes.
    Returned arrays have shape (size, size).
    """
    lin = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(lin, lin)
    return x, y


class HoloSimulator:
    """
    Final stable holographic simulator.
    Computes wave interference patterns from Symbol objects.
    """

    def __init__(self):
        pass

    def compute_field(self, x, y, symbols):
        """
        Compute a complex holographic interference field.

        Parameters:
            x, y: 2D grids (numpy arrays)
            symbols: list of Symbol objects

        Returns:
            field: 2D numpy array (complex128)
        """

        field = np.zeros_like(x, dtype=np.complex128)

        for sym in symbols:
            dx = x - sym.x
            dy = y - sym.y
            r = np.sqrt(dx**2 + dy**2) + 1e-6  # avoid division by zero

            # BASE WAVE
            wave = sym.amplitude * np.exp(1j * sym.base_freq * r)

            # HARMONICS (multiplier, relative amp, phase)
            for (mult, rel, ph) in sym.harmonics:
                wave += sym.amplitude * rel * np.exp(
                    1j * (sym.base_freq * mult * r + ph)
                )

            field += wave

        return field
