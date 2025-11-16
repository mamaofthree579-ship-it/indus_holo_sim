# simulator/symbol.py
"""
Stable Symbol class used by the simulator and Streamlit UI.

Constructor matches the streamlit_app usage:
    Symbol(name=<str>, frequency=<float>, sigma=<float>, harmonics=<list>)

Harmonics normalized to list of (mult, rel_amp, phase) tuples.
"""

from typing import List, Tuple

class Symbol:
    def __init__(
        self,
        name: str,
        frequency: float = 25.0,
        sigma: float = 0.06,
        harmonics: List = None,
        amplitude: float = 1.0,
        x: float = 0.5,
        y: float = 0.5,
    ):
        self.name = name
        self.frequency = float(frequency)
        self.sigma = float(sigma)
        self.amplitude = float(amplitude)
        self.x = float(x)
        self.y = float(y)

        # Normalize harmonics into (mult, rel_amp, phase) tuples of floats
        if harmonics is None:
            self.harmonics = []
        else:
            processed = []
            for h in harmonics:
                if isinstance(h, (int, float)):
                    processed.append((float(h), 1.0, 0.0))
                elif isinstance(h, (list, tuple)):
                    mult = float(h[0])
                    rel = float(h[1]) if len(h) > 1 else 1.0
                    phase = float(h[2]) if len(h) > 2 else 0.0
                    processed.append((mult, rel, phase))
            self.harmonics = processed

    def __repr__(self):
        return (
            f"Symbol(name={self.name}, freq={self.frequency}, amp={self.amplitude}, "
            f"harmonics={self.harmonics}, pos=({self.x:.2f},{self.y:.2f}), sigma={self.sigma})"
        )
