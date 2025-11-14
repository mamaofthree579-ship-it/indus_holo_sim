# simulator/symbol.py

class Symbol:
    """
    Unified Symbol class – supports:
      - name
      - base_freq
      - harmonics
      - image_path  <-- REQUIRED BY STREAMLIT
      - amplitude
      - x, y

    This version is guaranteed compatible with streamlit_app.py.
    """

    def __init__(
        self,
        name: str,
        base_freq: float = 440.0,
        harmonics=None,
        image_path: str = None,
        amplitude: float = 1.0,
        x: float = 0.5,
        y: float = 0.5
    ):
        self.name = name
        self.base_freq = float(base_freq)
        self.image_path = image_path
        self.amplitude = float(amplitude)
        self.x = float(x)
        self.y = float(y)

        # Normalize harmonics
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
            f"Symbol(name={self.name}, base_freq={self.base_freq}, "
            f"harmonics={self.harmonics}, image_path={self.image_path}, "
            f"pos=({self.x}, {self.y}), amplitude={self.amplitude})"
        )
