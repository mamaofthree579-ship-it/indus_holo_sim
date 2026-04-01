import numpy as np

# ------------------------------------------------------------------------------
# Normalize Energy
# ------------------------------------------------------------------------------
def normalize_energy(values):
    values = np.array(values)
    if np.max(values) == 0:
        return values
    return values / np.max(values)


# ------------------------------------------------------------------------------
# Generate Harmonic Modulation
# ------------------------------------------------------------------------------
def harmonic_modulation(base_values, phase):
    """Apply sinusoidal modulation to simulate energy oscillation."""
    return base_values * (1 + 0.3 * np.sin(phase))


# ------------------------------------------------------------------------------
# Compute Phase Array
# ------------------------------------------------------------------------------
def compute_phase(n, t):
    return np.linspace(0, 2 * np.pi, n) + t
