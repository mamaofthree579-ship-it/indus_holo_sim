# simulator/symbol.py

class Symbol:
    """
    Final stable Symbol class for the Indus holographic simulator.

    Supports:
      - name
      - base_freq (float)
      - harmonics: list of (mult, rel_amp, phase)
      - amplitude
      - image_path
      - x,y placement (float)
    """

    def __init__(
        self,
        name: str,
        base_freq: float = 440.0,
        harmonics=None,
        image_path: str = None,
        amplitude: float = 1.0,
        x: float = 0.5,
        y: float = 0.5,
    ):
        self.name = name

        # Ensure all critical values are floats
        self.base_freq = float(base_freq)
        self.amplitude = float(amplitude)
        self.x = float(x)
        self.y = float(y)

        self.image_path = image_path

        # Normalize harmonics into tuples of floats
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
                else:
                    # fallback: ignore invalid harmonics
                    continue
            self.harmonics = processed

    def __repr__(self):
        return (
            f"Symbol(name={self.name}, freq={self.base_freq}, amp={self.amplitude}, "
            f"harmonics={self.harmonics}, pos=({self.x},{self.y}))"
        )
