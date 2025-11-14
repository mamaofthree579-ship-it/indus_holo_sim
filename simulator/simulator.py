def compute_field(self, x, y, symbols):
    field = np.zeros_like(x, dtype=np.complex128)

    for sym in symbols:
        dx = x - sym.x
        dy = y - sym.y
        r = np.sqrt(dx**2 + dy**2) + 1e-6

        # Base wave
        wave = sym.amplitude * np.exp(1j * sym.base_freq * r)

        # Harmonics stored as (mult, rel_amp, phase)
        for (mult, rel, ph) in sym.harmonics:
            wave += sym.amplitude * rel * np.exp(1j * (sym.base_freq * mult * r + ph))

        field += wave

    return field
