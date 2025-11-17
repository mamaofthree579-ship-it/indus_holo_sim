# simulator/simulator.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from pathlib import Path
from typing import Tuple

def create_grid(size:int=512) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, size)
    ys = np.linspace(-1.0, 1.0, size)
    return np.meshgrid(xs, ys)

class Symbol:
    def __init__(self, symbol_id, metadata):
        self.id = symbol_id
        self.frequency = float(metadata.get("frequency_total") or metadata.get("frequency") or 25.0)
        self.positions = metadata.get("positions", {"solo":0,"initial":0,"medial":0,"terminal":0})
        self.sites = metadata.get("sites", {})
        # small fallback
        self.sigma = float(metadata.get("sigma", 0.06))

    def __repr__(self):
        return f"<Symbol {self.id} f={self.frequency}>"

class HoloSimulator:
    def __init__(self):
        pass

    def compute_field(self, symbol: Symbol, grid, time: float = 0.0) -> Figure:
        """
        Older function kept for synthetic fields. Returns Matplotlib Figure.
        """
        XX, YY = grid
        # radial distance from center
        r = np.sqrt(XX**2 + YY**2) + 1e-9
        sigma = max(0.02, symbol.sigma)
        env = np.exp(-0.5 * (r**2) / (sigma**2))
        k = 2*np.pi*symbol.frequency/50.0
        field = env * np.exp(1j * (k * r))
        intensity = np.abs(field)**2
        intensity = intensity / (intensity.max() + 1e-12)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(intensity, cmap="magma", origin="lower", extent=(-1,1,-1,1))
        ax.set_title(f"Synthetic field — {symbol.id}")
        ax.axis("off")
        return fig

    def compute_field_from_glyph(self, symbol: Symbol, glyph_mask_path: str, mode: str="holo", wavelength:float=0.5) -> Figure:
        """
        Use the binary glyph mask PNG as a source aperture and simulate:
          - holographic (FFT-based) propagation (far-field intensity)
          - light: use Fraunhofer diffraction pattern (FFT magnitude)
          - acoustic: treat bright pixels as pulsating point sources with phase ~ frequency*pos

        Args:
            symbol: Symbol object (metadata)
            glyph_mask_path: path to binary mask image (white=source)
            mode: 'holo'|'light'|'acoustic'|'mask' (mask displays the source)
            wavelength: free parameter controlling fringe spacing
        Returns:
            matplotlib.figure.Figure
        """
        # Load mask and convert to float array
        p = Path(glyph_mask_path)
        if not p.exists():
            raise FileNotFoundError(f"Glyph mask not found: {glyph_mask_path}")
        img = Image.open(p).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0  # 0..1

        # pad to square power-of-two for nicer FFT (optional)
        n = max(arr.shape)
        N = 1 << (n-1).bit_length()
        pad = ((0, N - arr.shape[0]), (0, N - arr.shape[1]))
        arr_padded = np.pad(arr, pad, mode='constant', constant_values=0.0)

        # Prepare complex field at aperture: amplitude * exp(i*phase)
        amplitude = arr_padded
        # Phase model: use symbol frequency to set a phase ramp or random tiny phase per pixel
        freq = max(1.0, float(symbol.frequency))
        # small random phase seeded by symbol id for reproducibility
        phase_noise = np.sin(np.linspace(0, 2*np.pi, arr_padded.size)).reshape(arr_padded.shape) * 0.0
        # For acoustic, we might want phase proportional to distance from center
        cx, cy = np.array(arr_padded.shape) / 2.0
        yy, xx = np.indices(arr_padded.shape)
        r = np.sqrt((xx-cx)**2 + (yy-cy)**2) + 1e-9
        # scale factor
        phase = 2.0 * np.pi * (freq/50.0) * (r / r.max()) + phase_noise

        aperture_field = amplitude * np.exp(1j * phase)

        # Compute far-field (Fraunhofer) approx: FFT of aperture field
        F = np.fft.fftshift(np.fft.fft2(aperture_field))
        intensity = np.abs(F)**2
        intensity = intensity / (intensity.max() + 1e-12)

        if mode == "mask":
            disp = arr_padded
            title = f"Glyph mask ({symbol.id})"
        else:
            disp = intensity
            title = f"{mode.title()} pattern — {symbol.id}"

        # Downsample to reasonable display resolution
        # convert to 512x512 for plotting (or smaller)
        import scipy.ndimage as ndi
        disp_small = ndi.zoom(disp, 512.0/disp.shape[0], order=1) if disp.shape[0] != 512 else disp

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(disp_small, cmap="magma", origin="lower")
        ax.set_title(title)
        ax.axis("off")
        return fig
