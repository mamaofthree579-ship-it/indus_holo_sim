import numpy as np

class Symbol:
    def __init__(
        self,
        name: str,
        base_freq: float = 440.0,
        harmonics=None,
        strength: float = 1.0,
        image_path: str = None,
        x: float = 0.0,
        y: float = 0.0
    ):
        """
        A symbol in the holographic simulation.

        Parameters
        ----------
        name : NB code or label
        base_freq : float
        harmonics : list of multipliers
        strength : amplitude multiplier
        image_path : for display in UI
        x, y : placement in grid (centered coords, -1..1)
        """
        self.name = name
        self.base_freq = base_freq

        if harmonics is None:
            harmonics = []
        self.harmonics = harmonics

        self.strength = float(strength)
        self.image_path = image_path

        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"Symbol(name={self.name}, f={self.base_freq}, harm={self.harmonics}, strength={self.strength})"
