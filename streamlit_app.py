import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io
import matplotlib.pyplot as plt

st.title("🔮 Symbol Tester — Light, Acoustic & Holographic Explorer")
st.write("Upload a symbol image to test holographic, acoustic, and frequency responses.")

# ---------------------------------------------------------
# File upload
# ---------------------------------------------------------
uploaded = st.file_uploader("Upload Symbol Image (PNG/JPG)", 
                            type=["png", "jpg", "jpeg"])

if not uploaded:
    st.stop()

# load + normalize to 0–1
img = Image.open(uploaded).convert("L")
img = ImageOps.invert(img)  # ink = bright
img_np = np.array(img) / 255.0

st.subheader("Uploaded Symbol")
st.image(img, use_column_width=True)

# ---------------------------------------------------------
# Simulation Controls
# ---------------------------------------------------------

st.sidebar.title("Simulation Controls")

mode = st.sidebar.radio(
    "Mode",
    ["Light FFT (Diffraction)", "Acoustic Harmonics", "Holographic Interference"]
)

# shared settings
resize = st.sidebar.slider("Resize Image (px)", 64, 512, 256)
img_resized = img.resize((resize, resize))
img_np = np.array(img_resized) / 255.0


# ---------------------------------------------------------
# 1. Light FFT Diffraction Simulation
# ---------------------------------------------------------
def simulate_fft(image):
    fft = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.abs(fft)
    magnitude = magnitude ** 0.4  # perceptual mapping
    magnitude /= magnitude.max() + 1e-8
    return magnitude

# ---------------------------------------------------------
# 2. Acoustic Harmonic Response
# ---------------------------------------------------------
def simulate_acoustic(image, base_freq, harmonics):
    h, w = image.shape
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(x, y)

    field = np.zeros_like(image, dtype=float)
    for k in range(1, harmonics + 1):
        field += image * np.sin(2 * np.pi * (base_freq * k) * (xx + yy))

    field = np.abs(field)
    field /= field.max() + 1e-8
    return field

# ---------------------------------------------------------
# 3. Holographic Interference (2-beam)
# ---------------------------------------------------------
def simulate_hologram(image, wavelength, angle):
    h, w = image.shape
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(x, y)

    carrier = np.cos(2 * np.pi * (xx*np.cos(angle) + yy*np.sin(angle)) / wavelength)

    holo = (image * carrier)
    holo = holo - holo.min()
    holo /= holo.max() + 1e-8
    return holo

# ---------------------------------------------------------
# Run Simulation
# ---------------------------------------------------------
if mode == "Light FFT (Diffraction)":
    st.header("🔆 Light Diffraction (FFT)")
    result = simulate_fft(img_np)

    fig, ax = plt.subplots()
    ax.imshow(result, cmap="inferno")
    ax.axis("off")
    st.pyplot(fig)


elif mode == "Acoustic Harmonics":
    st.header("🔊 Acoustic Harmonic Response")

    base = st.sidebar.slider("Base Frequency", 1, 20, 5)
    harms = st.sidebar.slider("Harmonics", 1, 20, 5)

    result = simulate_acoustic(img_np, base, harms)

    fig, ax = plt.subplots()
    ax.imshow(result, cmap="viridis")
    ax.axis("off")
    st.pyplot(fig)


elif mode == "Holographic Interference":
    st.header("🌐 Holographic Interference Pattern")

    wavelength = st.sidebar.slider("Wavelength", 0.01, 0.20, 0.05)
    angle_deg = st.sidebar.slider("Beam Angle (degrees)", 0, 180, 45)
    angle = np.deg2rad(angle_deg)

    result = simulate_hologram(img_np, wavelength, angle)

    fig, ax = plt.subplots()
    ax.imshow(result, cmap="magma")
    ax.axis("off")
    st.pyplot(fig)

# ---------------------------------------------------------
# Download Simulation Output
# ---------------------------------------------------------
buf = io.BytesIO()
out_img = Image.fromarray((result * 255).astype(np.uint8))
out_img.save(buf, format="PNG")
buf.seek(0)

st.download_button(
    "Download Simulation Output",
    buf,
    file_name=f"{mode.replace(' ', '_')}.png",
    mime="image/png"
)
