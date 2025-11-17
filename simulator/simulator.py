# simulator/simulator.py

import json
import numpy as np
from pathlib import Path

def safe_int(value, default=0):
    """Convert None to default, pass through ints, ignore bad types."""
    if value is None:
        return default
    try:
        return int(value)
    except:
        return default

class Symbol:
    def __init__(self, symbol_id, metadata):
        self.id = symbol_id

        # Academic metadata
        self.icit_id = metadata.get("icit_id")
        self.sign_class = metadata.get("class")
        self.set = metadata.get("set")
        self.frequency = safe_int(metadata.get("frequency"), default=1)

        pos = metadata.get("positions", {})
        self.positions = {
            "solo": safe_int(pos.get("solo")),
            "initial": safe_int(pos.get("initial")),
            "medial": safe_int(pos.get("medial")),
            "terminal": safe_int(pos.get("terminal"))
        }

        sites = metadata.get("sites", {})
        self.sites = {k: safe_int(v) for k, v in sites.items()}

    def generate_wave_matrix(self, mode="acoustic"):
        """
        Generates a holographic matrix with robust handling for null values.
        """
        base_freq = max(1, self.frequency)

        pos_factor = (
            self.positions["initial"]
            + self.positions["medial"]
            + self.positions["terminal"]
            + self.positions["solo"]
        )
        if pos_factor == 0:
            pos_factor = 1

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
    m = matrix - matrix.min()
    if m.max() > 0:
        m = m / m.max()
    return m
