# simulator/symbol.py

import numpy as np

class Symbol:
    """
    Lightweight symbol object used by the simulator and UI.
    No 'strength' kwarg — keep it simple and robust.

    Fields:
      - name: label (e.g. "NB023")
      - base_freq: base frequency in Hz
      - harmonics: list of multipliers or tuples (mult, rel_amp, phase)
      - image_path: optional path to a glyph thumbnail
      - x,y: position in normalized coords (0..1)
      - amplitude: overall amplitude multiplier (optional)
    """

    def __init__(
        self,
        name: str,
        base_freq: float = 440.0,
        harmonics=None,
        image_path: str = None,
        x: float = 0.5,
        y: float = 0.5,
        amplitude: float = 1.0
    ):
        self.name = name
        self.base_freq = float(base_freq)
        # harmonics may be either [mult1, mult2,...] or [(mult,rel,phase),...]
        if harmonics is None:
            self.harmonics = []
        else:
            # normalize simple numeric multipliers into tuple form
            normalized = []
            for h in harmonics:
                if isinstance(h, (int, float)):
                    normalized.append((float(h), 1.0, 0.0))
                elif isinstance(h, (list, tuple)) and len(h) >= 1:
                    # accept (mult,) or (mult,rel,phase)
                    mult = float(h[0])
                    rel = float(h[1]) if len(h) >= 2 else 1.0
                    ph  = float(h[2]) if len(h) >= 3 else 0.0
                    normalized.append((mult, rel, ph))
                else:
                    # fallback
                    normalized.append((1.0, 1.0, 0.0))
            self.harmonics = normalized

        self.image_path = image_path
        self.x = float(x)
        self.y = float(y)
        self.amplitude = float(amplitude)

    def __repr__(self):
        return f"Symbol({self.name}, f={self.base_freq}, harm={self.harmonics}, pos=({self.x:.2f},{self.y:.2f}))"
