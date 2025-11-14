import numpy as np
from simulator.symbol import Symbol


# -----------------------------------------------------
# Create coordinate grid
# -----------------------------------------------------
def create_grid(size: int):
    """
    Create centered 2D grid scaled -1..1
    """
    lin = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(lin, lin)
    return x, y


# -----------------------------------------------------
# Holographic Simulator
# -----------------------------------------------------
class HoloSimulator:
    def __init__(self):
        pass

    def compute_field(self, x, y, symbols):
        """
        Sum complex wavefields emitted by each symbol.
        """
        field = np.zeros_like(x, dtype=np.complex128)

        for sym in symbols:
            dx = x - sym.x
            dy = y - sym.y
            r = np.sqrt(dx**2 + dy**2) + 1e-6

            # base wave
            wave = np.exp(1j * sym.base_freq * r)

            # harmonics
            for h in sym.harmonics:
                wave += np.exp(1j * sym.base_freq * h * r)

            field += sym.strength * wave

        return field
