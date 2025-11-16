import json
import numpy as np
from pathlib import Path

class Symbol:
    def __init__(self, symbol_id, metadata):
        self.id = symbol_id

        # Academic metadata (new)
        self.icit_id = metadata.get("icit_id")
        self.sign_class = metadata.get("class")
        self.set = metadata.get("set")
        self.frequency = metadata.get("frequency")

        self.positions = metadata.get("positions", {
            "solo": 0,
            "initial": 0,
            "medial": 0,
            "terminal": 0
        })

        self.sites = metadata.get("sites", {})

        # Old fields kept optional (simulator safety)
        self.harmonics = metadata.get("harmonics", [])
        self.wave_class = metadata.get("wave_class", None)

    def generate_wave_matrix(self, mode="acoustic"):
        """
        Generates a numerical 'holographic' matrix for visualization
        based on sign metadata (frequency, positional balance, etc.)
        """
        base_freq = self.frequency if self.frequency else 1

        pos_factor = (
            self.positions.get("initial", 0)
            + self.positions.get("medial", 0)
            + self.positions.get("terminal", 0)
            + self.positions.get("solo", 0)
        ) or 1

        size = int(min(64, max(16, base_freq % 50 + 16)))

        if mode == "acoustic":
            x = np.linspace(0, np.pi * 4, size)
            y = np.linspace(0, np.pi * 4, size)
            X, Y = np.meshgrid(x, y)
            return np.sin(X * base_freq / 50) * np.sin(Y * pos_factor / 50)

        elif mode == "light":
            x = np.linspace(0, np.pi * 8, size)
            y = np.linspace(0, np.pi * 8, size)
            X, Y = np.meshgrid(x, y)
            return np.cos(X * base_freq / 80) * np.sin(Y * pos_factor / 80)

        elif mode == "matrix":
            return np.random.rand(size, size)

        return np.zeros((32, 32))


def load_signs_json(path="data/nb_signs.json"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find sign file: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return {k: Symbol(k, v) for k, v in data.items()}


def create_grid(matrix):
    """ Convert matrix into a normalized grayscale grid for plotting. """
    m = matrix - matrix.min()
    if m.max() > 0:
        m = m / m.max()
    return m
