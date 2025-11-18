# pages/4_Symbol_Tester_Advanced.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io
import matplotlib.pyplot as plt
import base64
from typing import List, Tuple

st.set_page_config(layout="wide", page_title="Indus Symbol Tester — Advanced")
st.title("Indus Symbol Tester — Advanced (light, acoustic, holography, analytics)")

# ---------------------------
# Helpers: image preprocessing
# ---------------------------
def load_and_preprocess(file, size=256, invert=True):
    img = Image.open(file).convert("L")
    # crop tiny borders, autocontrast
    img = ImageOps.autocontrast(img)
    if invert:
        img = ImageOps.invert(img)
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=float) / 255.0
    return arr, img

def array_to_pil(arr):
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(a)

# ---------------------------
# FFT / diffraction
# ---------------------------
def simulate_fft(image: np.ndarray, gamma=0.4):
    F = np.fft.fftshift(np.fft.fft2(image))
    mag = np.abs(F)
    mag = mag ** gamma
    mag /= (mag.max() + 1e-12)
    return mag

# ---------------------------
# Acoustic harmonic summation
# ---------------------------
def simulate_acoustic(image: np.ndarray, base_freq=5.0, harmonics=6):
    h, w = image.shape
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(x, y)
    field = np.zeros_like(image, dtype=float)
    for k in range(1, harmonics + 1):
        field += image * np.sin(2 * np.pi * base_freq * k * (xx + yy))
    field = np.abs(field)
    field /= (field.max() + 1e-12)
    return field

# ---------------------------
# Multi-beam holographic interference (Fraunhofer-like)
# Each beam: (angle radians, wavelength, amplitude, phase)
# ---------------------------
def simulate_hologram_multi(image: np.ndarray, beams: List[Tuple[float,float,float,float]]):
    # treat bright pixels as aperture amplitude
    amp = image.copy()
    h, w = amp.shape
    y = np.linspace(-0.5, 0.5, h)
    x = np.linspace(-0.5, 0.5, w)
    xx, yy = np.meshgrid(x, y)
    total_field = np.zeros_like(amp, dtype=complex)
    for (angle, wavelength, amplitude, phase) in beams:
        # plane-wave carrier for this beam
        k = 2*np.pi / max(1e-6, wavelength)
        carrier = np.exp(1j * (k * (xx * np.cos(angle) + yy * np.sin(angle)) + phase))
        field = amplitude * amp * carrier
        # far-field via FFT of field (aperture->far-field)
        F = np.fft.fftshift(np.fft.fft2(field))
        total_field += F
    intensity = np.abs(total_field)**2
    intensity = intensity / (intensity.max() + 1e-12)
    return intensity

# ---------------------------
# Activation map (local energy via integral image)
# ---------------------------
def integral_image(arr):
    return arr.cumsum(axis=0).cumsum(axis=1)

def local_sum_from_integral(I, x0,y0,x1,y1):
    # inclusive coordinates
    s = I[y1, x1]
    if x0>0: s -= I[y1, x0-1]
    if y0>0: s -= I[y0-1, x1]
    if x0>0 and y0>0: s += I[y0-1, x0-1]
    return s

def activation_map(field, window=15):
    # field: energy map (non-negative)
    A = field.astype(float)
    I = integral_image(A)
    H,W = A.shape
    out = np.zeros_like(A)
    r = window//2
    for y in range(H):
        y0 = max(0, y - r)
        y1 = min(H-1, y + r)
        for x in range(W):
            x0 = max(0, x - r)
            x1 = min(W-1, x + r)
            out[y,x] = local_sum_from_integral(I, x0,y0,x1,y1)
    out /= (out.max()+1e-12)
    return out

# ---------------------------
# Fourier descriptors (boundary-based)
# ---------------------------
def fourier_descriptors(image: np.ndarray, n_coeffs=64):
    # binary edge detection: sobel-like via np.gradient
    gx, gy = np.gradient(image)
    grad = np.hypot(gx, gy)
    thresh = np.percentile(grad, 70)
    edges = (grad > thresh).astype(np.uint8)
    ys, xs = np.nonzero(edges)
    if len(xs) < 8:
        # fallback: sample outer contour by thresholding
        ys, xs = np.nonzero(image > (image.mean()*0.5))
    pts = np.column_stack((xs, ys))
    if pts.shape[0] < 8:
        # fallback -> empty descriptor
        return np.zeros(n_coeffs)
    # center and make complex sequence
    complex_pts = pts[:,0] + 1j * pts[:,1]
    # resample to uniform length by interpolation (simple)
    L = len(complex_pts)
    idx = np.linspace(0, L-1, n_coeffs).astype(int)
    seq = complex_pts[idx]
    # DFT and take magnitude (skip DC)
    fd = np.fft.fft(seq)
    mag = np.abs(fd)
    mag = mag / (mag.max() + 1e-12)
    return mag[:n_coeffs//2]  # return half-spectrum

# ---------------------------
# Similarity scoring
# ---------------------------
def descriptor_distance(d1, d2):
    # L2 normalized
    d1 = np.array(d1)/ (np.linalg.norm(d1)+1e-12)
    d2 = np.array(d2)/ (np.linalg.norm(d2)+1e-12)
    return np.linalg.norm(d1 - d2)

def fft_correlation(a,b):
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    # use normalized cross-correlation of magnitudes
    Am = np.abs(A); Bm = np.abs(B)
    Am /= (Am.max()+1e-12); Bm /= (Bm.max()+1e-12)
    return np.sum(Am * Bm) / (np.sqrt(np.sum(Am**2)*np.sum(Bm**2))+1e-12)

# ---------------------------
# UI: Upload multiple symbols
# ---------------------------
st.sidebar.header("Uploads & Global Settings")
uploaded_files = st.sidebar.file_uploader("Upload 1–6 symbol images", accept_multiple_files=True, type=["png","jpg","jpeg"])
size = st.sidebar.slider("Working size (px)", 64, 512, 256)
gamma = st.sidebar.slider("FFT gamma (display)", 0.2, 1.5, 0.4)

if not uploaded_files:
    st.info("Upload one or more symbol PNG/JPG files in the sidebar to start.")
    st.stop()

# preprocess all uploads
symbols = []
pil_images = []
for f in uploaded_files[:6]:
    arr, pil = load_and_preprocess(f, size=size, invert=True)
    symbols.append(arr)
    pil_images.append(pil)

# display uploaded symbols
st.subheader("Uploaded Symbols")
cols = st.columns(len(symbols))
for i, img in enumerate(pil_images):
    with cols[i]:
        st.image(img, caption=f"Symbol {i+1}", use_column_width=True)

# ---------------------------
# Mode selection
# ---------------------------
st.sidebar.header("Simulation Mode")
mode = st.sidebar.selectbox("Mode", [
    "Light FFT (Diffraction)", 
    "Acoustic Harmonics", 
    "Multi-Beam Holography",
    "Activation Map",
    "Fourier Descriptors & Similarity",
    "Combine Symbols (co-activation)"
])

# ---------------------------
# Run chosen mode
# ---------------------------
result = None
if mode == "Light FFT (Diffraction)":
    st.header("🔆 Light Diffraction (FFT)")
    # apply to each symbol; show gallery
    figs = []
    for i, sym in enumerate(symbols):
        res = simulate_fft(sym, gamma=gamma)
        result = res
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(res, cmap="inferno")
        ax.set_title(f"FFT: Symbol {i+1}")
        ax.axis("off")
        st.pyplot(fig)

elif mode == "Acoustic Harmonics":
    st.header("🔊 Acoustic Harmonics (per-symbol)")
    base = st.sidebar.slider("Base frequency", 1, 20, 5)
    harms = st.sidebar.slider("Harmonics", 1, 20, 6)
    for i, sym in enumerate(symbols):
        res = simulate_acoustic(sym, base, harms)
        result = res
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(res, cmap="viridis")
        ax.set_title(f"Acoustic: Symbol {i+1}")
        ax.axis("off")
        st.pyplot(fig)

elif mode == "Multi-Beam Holography":
    st.header("🌐 Multi-Beam Holography (combine beams)")
    st.write("Each uploaded symbol will be used as an aperture for its own set of beams. You can configure up to 3 beams per symbol.")
    beams_per_symbol = []
    n_beams = st.sidebar.slider("Beams per symbol", 1, 3, 2)
    for idx in range(len(symbols)):
        st.subheader(f"Symbol {idx+1} beams")
        bs = []
        for b in range(n_beams):
            angle = st.slider(f"Symbol{idx+1} Beam{b+1} angle (deg)", 0, 360, int(45 + 30*b), key=f"ang_{idx}_{b}")
            wl = st.slider(f"Symbol{idx+1} Beam{b+1} wavelength", 0.01, 0.5, 0.05, key=f"wl_{idx}_{b}")
            amp = st.slider(f"Symbol{idx+1} Beam{b+1} amplitude", 0.0, 2.0, 1.0, key=f"amp_{idx}_{b}")
            phase = st.slider(f"Symbol{idx+1} Beam{b+1} phase (deg)", 0, 360, 0, key=f"phase_{idx}_{b}")
            bs.append((np.deg2rad(angle), wl, amp, np.deg2rad(phase)))
        beams_per_symbol.append(bs)

    # compute far-field per symbol then sum across symbols
    total_field = None
    for idx, sym in enumerate(symbols):
        intensity = simulate_hologram_multi(sym, beams_per_symbol[idx])
        if total_field is None:
            total_field = intensity
        else:
            total_field += intensity
    total_field = total_field / (total_field.max()+1e-12)
    result = total_field
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(total_field, cmap="magma")
    ax.set_title("Combined multi-symbol holographic intensity")
    ax.axis("off")
    st.pyplot(fig)

elif mode == "Activation Map":
    st.header("📈 Activation Map (where energy concentrates)")
    choice = st.selectbox("Choose symbol to map", [f"Symbol {i+1}" for i in range(len(symbols))])
    idx = int(choice.split()[1]) - 1
    # compute FFT intensity as base field
    base_field = simulate_fft(symbols[idx], gamma=gamma)
    act = activation_map(base_field, window=st.sidebar.slider("Activation window", 3, 61, 15))
    result = act
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(act, cmap="inferno")
    ax.set_title(f"Activation map — {choice}")
    ax.axis("off")
    st.pyplot(fig)

elif mode == "Fourier Descriptors & Similarity":
    st.header("🔬 Fourier Descriptors & Similarity")
    n_coeffs = st.sidebar.slider("Descriptor length", 8, 256, 64)
    descriptors = []
    for i, sym in enumerate(symbols):
        fd = fourier_descriptors(sym, n_coeffs)
        descriptors.append(fd)
        st.write(f"Symbol {i+1} descriptor (first 8):", list(np.round(fd[:8], 3)))
    # pairwise distances
    st.write("Pairwise descriptor distances (lower = more similar)")
    n = len(descriptors)
    dist_mat = np.zeros((n,n))
    corr_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dist_mat[i,j] = descriptor_distance(descriptors[i], descriptors[j])
            corr_mat[i,j] = fft_correlation(symbols[i], symbols[j])
    st.write("Descriptor distance matrix:")
    st.dataframe(np.round(dist_mat, 4))
    st.write("FFT magnitude correlation matrix (higher = more similar):")
    st.dataframe(np.round(corr_mat, 4))
    # show best match for each
    for i in range(n):
        other = int(np.argmin([dist_mat[i,j] if i!=j else np.inf for j in range(n)]))
        st.write(f"Symbol {i+1} best descriptor match -> Symbol {other+1} (dist {dist_mat[i,other]:.3f})")

elif mode == "Combine Symbols (co-activation)":
    st.header("🔗 Combine symbols (sum apertures and examine fields)")
    # sum symbol arrays
    summed = np.zeros_like(symbols[0])
    for s in symbols:
        summed += s
    summed /= (summed.max()+1e-12)
    # choose simulation type for summed
    sim_type = st.selectbox("Simulate type for combined aperture", ["FFT", "Acoustic", "Holography"])
    if sim_type == "FFT":
        res = simulate_fft(summed, gamma=gamma)
    elif sim_type == "Acoustic":
        base = st.sidebar.slider("Base frequency", 1, 20, 5)
        harms = st.sidebar.slider("Harmonics", 1, 20, 6)
        res = simulate_acoustic(summed, base, harms)
    else:
        # simple single beam holography for combined aperture
        angle = np.deg2rad(st.sidebar.slider("Beam angle", 0, 360, 45))
        wl = st.sidebar.slider("Wavelength", 0.01, 0.5, 0.05)
        res = simulate_hologram_multi(summed, [(angle, wl, 1.0, 0.0)])
    result = res
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(res, cmap="magma")
    ax.axis("off")
    st.pyplot(fig)

# ---------------------------
# Download resulting image
# ---------------------------
if result is not None:
    out_pil = array_to_pil(result)
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Download Result PNG", data=buf, file_name="symbol_result.png", mime="image/png")

st.write("---")
st.caption("Notes: these are computational emulations — Fraunhofer (FFT) approximations for far-field patterns; acoustic model is a simplified phase-summed 2D model; Fourier descriptors are simple boundary-based descriptors for structural similarity. Use the sliders to explore parameter spaces. If you want Gerchberg–Saxton phase-holography, time-domain acoustic WAV export, or vector-based SVG rasterization hop next — I can add those.")
