import numpy as np
from simulator.symbol import Symbol

class HoloSimulator:
    """
    Computes holographic interference fields by superposing contributions
    from all Symbol nodes across time.
    """

    def __init__(self, XX, YY, time_samples=60, t_max=1.0):
        self.XX = XX
        self.YY = YY
        self.time_samples = time_samples
        self.t_max = t_max
        self.symbols = []

    def add_symbol(self, symbol):
        self.symbols.append(symbol)

    def clear(self):
        self.symbols = []

    def simulate(self):
        times = np.linspace(0, self.t_max, self.time_samples)
        total = np.zeros((self.time_samples, self.XX.shape[0], self.XX.shape[1]), dtype=np.float32)

        for s in self.symbols:
            total += s.contribution(times, self.XX, self.YY)

        intensity = np.mean(total**2, axis=0)
        intensity /= intensity.max() + 1e-12
        return intensity
