import numpy as np
from math import pi

C_PROP = 1.2       # propagation speed (abstract units)
ALPHA_ATTEN = 8.0  # attenuation constant

class Symbol:
    """
    Represents a holographic-frequency glyph node.
    Includes:
    - base frequency
    - harmonic multipliers
    - propagation delay
    - distance-based attenuation
    - Gaussian spatial envelope
    """

    def __init__(self, name, x, y, base_freq, amplitude=1.0, harmonics=None, sigma=0.06):
        self.name = name
        self.x = x
        self.y = y
        self.base_freq = base_freq
        self.amplitude = amplitude
        self.sigma = sigma

        if harmonics is None:
            self.harmonics = [(1.0, 1.0, 0.0)]  # (multiplier, relative_amp, phase_offset)
        else:
            self.harmonics = harmonics

    def distance(self, XX, YY):
        return np.sqrt((XX - self.x)**2 + (YY - self.y)**2)

    def envelope(self, dist):
        return np.exp(-(dist**2) / (2 * self.sigma**2))

    def attenuation(self, dist):
        return 1.0 / (1 + ALPHA_ATTEN * dist**2)

    def contribution(self, times, XX, YY):
        dist = self.distance(XX, YY)
        env = self.envelope(dist)
        att = self.attenuation(dist)

        field = np.zeros((len(times), XX.shape[0], XX.shape[1]), dtype=np.float32)

        for mult, rel_amp, phase_offset in self.harmonics:
            freq = self.base_freq * mult

            # phase shift from propagation delay
            phase_delay = 2 * pi * freq * (dist / C_PROP)

            for i, t in enumerate(times):
                field[i] += (
                    self.amplitude * rel_amp *
                    np.sin(2 * pi * freq * t + phase_offset - phase_delay) *
                    env * att
                )

        return field
