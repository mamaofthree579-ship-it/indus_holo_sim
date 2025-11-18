# pages/7_Diffraction_Lab.py

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide", page_title="Diffraction Lab")

st.title("🔆 Diffraction Lab — Fresnel, FFT, Interference")

st.markdown("""
This lab provides deeper control over diffraction simulation using:

- **Fresnel propagation** (near-to-mid-field)
- **Fraunhofer (FFT) diffraction** 
- **Symbol-preserving diffraction rendering**
- **Multi-symbol interference** (overlayed apertures)
- Adjustable wavelength, distance, and frequency space filtering

Useful for studying **ancient optical symbols**, coherence signatures,  
and light-field behavior of Indus signs under hypothetical illumination.
""")

# ------------------------------------------------------
# Utility functions
# ------------------------------------------------------

def load_symbol(file, size=256, invert=True):
    img = Image.open(file).convert("L")
    img = ImageOps.autocontrast(img)
    if invert:
        img = ImageOps.invert(img)
    img = img.resize((size, size))
    arr = np.array(img).astype(float) / 255.0
    return arr, img

def simulate_fft(image):
    F = np.fft.fftshift(np.fft.fft2(image))
    mag = np.abs(F)
    mag = mag ** 0.4
    mag /= mag.max() + 1e-12
    return mag

def simulate_fresnel(image, distance=1.0, wavelength=0.00065, lowpass=0.25, blend=0.3):
    img = image.astype(float)
    img /= img.max() + 1e-12

    N = img.shape[0]
    x = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, x)

    fx = np.fft.fftfreq(N, d=1/N)
    fy = np.fft.fftfreq(N, d=1/N)
    FX, FY = np.meshgrid(fx, fy)

    H = np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))

    U1 = np.fft.fft2(img)
    U2 = U1 * H

    r = np.sqrt(FX**2 + FY**2)
    LP = (r < lowpass).astype(float)
    U2 *= LP

    field = np.fft.ifft2(U2)
    intensity = np.abs(field)
    intensity /= intensity.max() + 1e-12

    blended = (1 - blend) * intensity + blend * img
    blended /= blended.max() + 1e-12

    return blended, intensity

# ------------------------------------------------------
# Sidebar
# ------------------------------------------------------

st.sidebar.header("Upload Symbols")

uploaded = st.sidebar.file_uploader(
    "Upload 1–3 Symbol Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

size = st.sidebar.slider("Resize to (px)", 64, 512, 256)

if not uploaded:
    st.info("Upload symbol images using the sidebar to begin.")
    st.stop()

symbols = []
pil_symbols = []

for f in uploaded[:3]:
    arr, pil = load_symbol(f, size=size, invert=True)
    symbols.append(arr)
    pil_symbols.append(pil)

st.subheader("Uploaded Symbols")
cols = st.columns(len(symbols))
for i, img in enumerate(pil_symbols):
    with cols[i]:
        st.image(img, caption=f"Symbol {i+1}", use_column_width=True)

# ------------------------------------------------------
# Mode selection
# ------------------------------------------------------

mode = st.sidebar.radio(
    "Simulation Mode",
    [
        "Enhanced Fresnel Diffraction",
        "Pure FFT (Fraunhofer) Diffraction",
        "Multi-Symbol Interference (Fresnel)",
    ]
)

# ------------------------------------------------------
# Enhanced Fresnel Mode
# ------------------------------------------------------

if mode == "Enhanced Fresnel Diffraction":
    st.header("Enhanced Fresnel Diffraction")

    distance = st.sidebar.slider("Propagation distance", 0.1, 3.0, 1.0)
    wavelength = st.sidebar.slider("Wavelength", 0.0001, 0.002, 0.00065)
    lowpass = st.sidebar.slider("Low-pass filter radius", 0.05, 1.0, 0.25)
    blend = st.sidebar.slider("Blend with original symbol", 0.0, 1.0, 0.35)

    for i, sym in enumerate(symbols):
        enhanced, raw = simulate_fresnel(sym, distance, wavelength, lowpass, blend)

        st.subheader(f"Symbol {i+1}")

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(raw, cmap="inferno")
        ax[0].set_title("Raw Fresnel Diffraction")
        ax[0].axis("off")

        ax[1].imshow(enhanced, cmap="magma")
        ax[1].set_title("Enhanced (Symbol-Preserving)")
        ax[1].axis("off")

        st.pyplot(fig)

# ------------------------------------------------------
# Pure FFT Diffraction
# ------------------------------------------------------

elif mode == "Pure FFT (Fraunhofer) Diffraction":
    st.header("Fraunhofer Diffraction (FFT)")

    for i, sym in enumerate(symbols):
        diff = simulate_fft(sym)

        st.subheader(f"Symbol {i+1}")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(diff, cmap="inferno")
        ax.set_title("FFT Diffraction")
        ax.axis("off")
        st.pyplot(fig)

# ------------------------------------------------------
# Multi-Symbol Interference
# ------------------------------------------------------

elif mode == "Multi-Symbol Interference (Fresnel)":
    st.header("Multi-Symbol Interference (Fresnel Propagation)")

    if len(symbols) < 2:
        st.warning("Upload 2 or 3 symbols for interference.")
        st.stop()

    distance = st.sidebar.slider("Propagation distance", 0.1, 3.0, 1.4)
    wavelength = st.sidebar.slider("Wavelength", 0.0001, 0.002, 0.00065)

    summed = np.zeros_like(symbols[0])
    for s in symbols:
        summed += s
    summed /= summed.max() + 1e-12

    enhanced, raw = simulate_fresnel(summed, distance, wavelength)

    st.subheader("Combined Symbols Interference")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(raw, cmap="inferno")
    ax[0].set_title("Raw Fresnel Interference")
    ax[0].axis("off")

    ax[1].imshow(enhanced, cmap="magma")
    ax[1].set_title("Enhanced Interference")
    ax[1].axis("off")

    st.pyplot(fig)
