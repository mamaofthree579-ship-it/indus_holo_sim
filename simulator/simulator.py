# simulator/simulator.py

import numpy as np
from simulator.symbol import Symbol

def create_grid(size: int):
    """
    Create a normalized 0..1 grid of shape (size,size).
    """
    X = np.linspace(0.0, 1.0, size)
    Y = np.linspace(0.0, 1.0, size)
    XX, YY = np.meshgrid(X, Y)
    return XX, YY


class HoloSimulator:
    """
    Computes holographic interference fields from Symbol objects.
    """

    def __init__(self):
        pass

    def compute_field(self, XX, YY, symbols, times=None):
        """
        Compute either instantaneous complex field (if times is scalar or list of t)
        or time-averaged intensity (if times is array/list -> averaged magnitude^2).

        Args:
            XX, YY: 2D coordinate grids (0..1)
            symbols: list of Symbol objects
            times: None -> returns instantaneous field at t=0
                   scalar -> instantaneous at t
                   1D array -> compute time-average intensity over times

        Returns:
            If times is None or scalar -> complex field 2D array
            If times is 1D array -> real intensity 2D array normalized 0..1
        """
        # treat coords: convert normalized 0..1 to centered coords with physical scale
        # choose a scale so that distance ~ [0..sqrt(2)]
        Xc = XX
        Yc = YY

        if times is None:
            times = np.array([0.0], dtype=float)
            time_mode = "instant"
        elif np.isscalar(times):
            times = np.array([float(times)], dtype=float)
            time_mode = "instant"
        else:
            times = np.array(times, dtype=float)
            time_mode = "avg"

        nt = len(times)
        grid_shape = XX.shape
        total_field = np.zeros((nt, grid_shape[0], grid_shape[1]), dtype=np.complex128)

        for s in symbols:
            # ensure symbol parameters are floats
            sx = float(s.x)
            sy = float(s.y)
            amp = float(s.amplitude)
            freq0 = float(s.base_freq)
            sigma = float(s.sigma)

            dx = Xc - sx
            dy = Yc - sy
            r = np.sqrt(dx * dx + dy * dy) + 1e-6  # avoid zero

            # spatial envelope (gaussian)
            env = np.exp(- (r**2) / (2 * sigma * sigma))

            # time loop for symbol contributions
            for ti_idx, t in enumerate(times):
                # start with base harmonic
                field_t = amp * env * np.exp(1j * (2.0 * np.pi * freq0 * t - r * 2.0 * np.pi * freq0))
                # add harmonics (mult, rel, phase)
                for (mult, rel, ph) in s.harmonics:
                    f = freq0 * float(mult)
                    field_t += amp * rel * env * np.exp(1j * (2.0 * np.pi * f * t + ph - r * 2.0 * np.pi * f))
                total_field[ti_idx] += field_t

        if time_mode == "instant":
            return total_field[0]
        else:
            # compute time-averaged intensity <|field|^2>_t and normalize
            intensity = np.mean(np.abs(total_field)**2, axis=0)
            # normalize
            intensity /= (intensity.max() + 1e-12)
            return intensity
