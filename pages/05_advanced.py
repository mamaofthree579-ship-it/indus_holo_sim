# pages/5_Advanced_Extras.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import matplotlib.pyplot as plt
import wave
import struct
import math

st.set_page_config(layout="wide")
st.title("Advanced Extras — Gerchberg–Saxton, WAV Export, & Fourier Descriptors")

st.markdown("""
This page provides 3 advanced tools:
1. **Gerchberg–Saxton** phase-hologram synthesis (phase-only SLM output).  
2. **Acoustic WAV export**: convert the 2D acoustic field into a mono WAV sample.  
3. **Improved Fourier descriptors** with rotation + scale invariance and contour overlay + similarity.
""")

# --------------------
# Utilities
# --------------------
def load_image_to_array(uploaded, size=256, invert=True):
    img = Image.open(uploaded).convert("L")
    img = ImageOps.autocontrast(img)
    if invert:
        img = ImageOps.invert(img)
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=float) / 255.0
    return arr, img

def array_to_image(arr):
    arrc = np.clip(arr, 0, 1)
    return Image.fromarray((arrc*255).astype(np.uint8))

# --------------------
# Gerchberg–Saxton implementation (CPU, numpy)
# --------------------
def gerchberg_saxon(target_amp, n_iters=100, tol=1e-6):
    """
    target_amp: 2D real array (0..1) desired far-field amplitude pattern.
    Returns: phase_only (radians) map same size as input (phase only SLM).
    Algorithm: standard Gerchberg–Saxton (iterate between target plane amplitude and aperture plane).
    Aperture plane initial guess: random phase with uniform amplitude 1 inside aperture.
    """
    M, N = target_amp.shape
    # desired far-field complex field (target amplitude, zero phase initially)
    TF = target_amp.astype(np.complex64)
    # initial aperture: uniform amplitude 1, random phase
    rng = np.random.default_rng(12345)
    aperture = np.exp(1j * 2*np.pi * rng.random((M,N)))
    last_error = None

    for k in range(n_iters):
        # forward: aperture -> far field
        F = np.fft.fft2(aperture)
        # replace amplitude with target amplitude, keep phase
        phase_F = np.angle(F)
        F_new = target_amp * np.exp(1j * phase_F)
        # backpropagate to aperture plane
        aperture_new = np.fft.ifft2(F_new)
        # enforce aperture amplitude = 1 (phase-only)
        aperture = np.exp(1j * np.angle(aperture_new))
        # compute error metric (L2 on amplitude)
        F_check = np.fft.fft2(aperture)
        err = np.linalg.norm(np.abs(F_check) - target_amp) / (np.linalg.norm(target_amp) + 1e-12)
        if last_error is not None and abs(last_error - err) < tol:
            break
        last_error = err

    # final SLM phase: angle of aperture
    phase_map = np.angle(aperture)
    # normalize to 0..2pi
    phase_map = np.mod(phase_map + 2*np.pi, 2*np.pi)
    return phase_map

# --------------------
# Acoustic WAV synthesis from 2D field
# --------------------
def synthesize_wav_from_field(field, duration_sec=2.0, sample_rate=22050, base_freq=440.0):
    """
    Convert a normalized 2D field into a mono audio waveform.
    Strategy:
      - collapse 2D field into 1D time-varying amplitude by summing columns weighted by a temporal carrier
      - or compute the dominant spatial frequencies and map to audio spectrum.
    Simpler approach: treat each column as a source with phase shift -> sum sinusoids.
    """
    H, W = field.shape
    # Normalize field
    F = field / (field.max() + 1e-12)
    t = np.linspace(0, duration_sec, int(duration_sec*sample_rate), endpoint=False)
    audio = np.zeros_like(t)
    # map W columns to slightly detuned sinusoids around base_freq
    for i in range(W):
        amp = F[:, i].mean()  # average over rows -> amplitude per column
        # detune frequency within +/- 10% depending on column index
        detune = (i - W/2) / (W/2) * 0.1
        freq = base_freq * (1.0 + detune)
        audio += amp * np.sin(2*np.pi*freq*t)
    # normalize to int16 range
    audio /= (np.max(np.abs(audio)) + 1e-12)
    audio_int16 = (audio * 0.9 * 32767).astype(np.int16)
    # write to bytes
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(audio_int16.tobytes())
    buf.seek(0)
    return buf

# --------------------
# Improved Fourier descriptors
# --------------------
def compute_fourier_descriptor(image, n_points=256):
    """
    Rotation & scale invariant Fourier descriptor:
      - binary threshold image
      - extract boundary coordinates (simple: sample coordinates of edge pixels)
      - resample sequence to n_points
      - compute complex sequence and take Fourier transform magnitudes (use magnitude => rotation invariance)
      - normalize by DC or overall norm for scale invariance
    Returns normalized magnitude descriptor (real vector).
    """
    # create edge map via gradient
    gy, gx = np.gradient(image)
    grad = np.hypot(gx, gy)
    # threshold gradient
    thr = np.percentile(grad, 60)
    edges = (grad > thr).astype(np.uint8)
    ys, xs = np.nonzero(edges)
    if len(xs) < 8:
        # fallback: sample points where image > mean
        ys, xs = np.nonzero(image > image.mean())
    if len(xs) == 0:
        return np.zeros(n_points//2)
    pts = np.column_stack((xs, ys)).astype(np.float64)
    # center the points
    pts -= pts.mean(axis=0, keepdims=True)
    # convert to complex sequence
    seq = pts[:,0] + 1j * pts[:,1]
    # resample to n_points uniformly (simple index interpolation)
    L = len(seq)
    idxs = (np.linspace(0, L-1, n_points)).astype(np.int32)
    seq_resampled = seq[idxs]
    # DFT
    fd = np.fft.fft(seq_resampled)
    mag = np.abs(fd)
    # discard DC (mag[0]) and keep half-spectrum for compactness
    mag = mag[1:(n_points//2)+1]
    # normalize
    mag /= (mag.max() + 1e-12)
    return mag

def descriptor_distance(d1, d2):
    d1 = np.array(d1); d2 = np.array(d2)
    # L2 distance after normalization
    dn = d1 / (np.linalg.norm(d1) + 1e-12)
    dm = d2 / (np.linalg.norm(d2) + 1e-12)
    return np.linalg.norm(dn - dm)

# --------------------
# UI layout
# --------------------
with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload symbol image (PNG/JPG)", type=['png','jpg','jpeg'])
    size = st.slider("Processing size (px)", 64, 512, 256)
    st.markdown("---")
    st.header("Choose tools")
    do_gs = st.checkbox("Gerchberg–Saxton (phase hologram)")
    do_wav = st.checkbox("Export WAV from acoustic field")
    do_fd  = st.checkbox("Fourier descriptors & similarity")

if not uploaded:
    st.info("Upload a symbol image (cropped) to enable tools.")
    st.stop()

# preprocess
arr, pil_img = load_image_to_array(uploaded, size=size, invert=True)
st.subheader("Input symbol")
st.image(pil_img, use_column_width=False)

# compute a base field to use for WAV generation (use FFT intensity)
base_fft = np.abs(np.fft.fft2(arr))
base_fft = np.fft.fftshift(base_fft)
base_fft /= (base_fft.max() + 1e-12)

# -------------------------------------------------
# Gerchberg–Saxton section
# -------------------------------------------------
if do_gs:
    st.subheader("Gerchberg–Saxton phase-hologram")
    n_iters = st.slider("GS iterations", 10, 1000, 200)
    # target amplitude set as the FFT magnitude of symbol (option)
    target_choice = st.selectbox("Target amplitude for GS", ["FFT magnitude (far-field)", "Symbol image (direct)"])
    if target_choice.startswith("FFT"):
        target = base_fft.copy()
        # normalize to 0..1
        target = (target - target.min()) / (target.max() - target.min() + 1e-12)
    else:
        target = arr.copy()
    st.write("Running Gerchberg–Saxton (this may take a few seconds)...")
    phase_map = gerchberg_saxon(target, n_iters=n_iters)
    # display phase map and simulated reconstruction
    fig1, ax1 = plt.subplots(1,2, figsize=(8,4))
    ax1[0].imshow(phase_map, cmap='twilight', vmin=0, vmax=2*np.pi)
    ax1[0].set_title("Phase-only SLM map (0..2π)")
    ax1[0].axis('off')
    # reconstruct far-field from phase map (assume unit aperture amplitude)
    aperture = np.exp(1j * phase_map)
    recon = np.abs(np.fft.fft2(aperture))
    recon = np.fft.fftshift(recon)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)
    ax1[1].imshow(recon, cmap='inferno')
    ax1[1].set_title("Simulated reconstruction (far-field)")
    ax1[1].axis('off')
    st.pyplot(fig1)
    # download phase map as image or numpy
    buf_img = io.BytesIO()
    Image.fromarray((phase_map / (2*np.pi) * 255).astype(np.uint8)).save(buf_img, format='PNG')
    buf_img.seek(0)
    st.download_button("Download phase map PNG", buf_img.getvalue(), file_name="phase_map.png", mime="image/png")
    # download numpy .npz
    npbuf = io.BytesIO()
    np.savez_compressed(npbuf, phase=phase_map)
    npbuf.seek(0)
    st.download_button("Download phase map (.npz)", npbuf, file_name="phase_map.npz", mime="application/octet-stream")

# -------------------------------------------------
# WAV export section
# -------------------------------------------------
if do_wav:
    st.subheader("Acoustic WAV export")
    duration = st.slider("Duration (s)", 0.5, 10.0, 2.0)
    sample_rate = st.selectbox("Sample rate", [8000,16000,22050,44100], index=2)
    base_freq = st.slider("Base frequency (Hz)", 100, 2000, 440)
    st.write("Synthesizing WAV from FFT intensity field...")
    wav_buf = synthesize_wav_from_field(base_fft, duration_sec=duration, sample_rate=sample_rate, base_freq=base_freq)
    st.audio(wav_buf)
    # provide download
    wav_buf.seek(0)
    st.download_button("Download WAV", wav_buf.read(), file_name="symbol_acoustic.wav", mime="audio/wav")

# -------------------------------------------------
# Fourier descriptor & similarity
# -------------------------------------------------
if do_fd:
    st.subheader("Fourier Descriptors & Similarity")
    n_coeffs = st.slider("Descriptor length (n_points)", 16, 1024, 256, step=16)
    fd = compute_fourier_descriptor(arr, n_points=n_coeffs)
    st.write("Descriptor (first 16 values):")
    st.write(np.round(fd[:16], 4))
    st.write("Descriptor length:", len(fd))
    # overlay contour on original
    # compute edge map
    gy, gx = np.gradient(arr)
    grad = np.hypot(gx, gy)
    thr = np.percentile(grad, 60)
    edges = (grad > thr).astype(np.uint8)
    ys, xs = np.nonzero(edges)
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].imshow(arr, cmap='gray')
    ax[0].scatter(xs, ys, s=1, c='r')
    ax[0].set_title("Symbol with detected edges")
    ax[0].axis('off')
    # show reconstructed descriptor-based shape (inverse FFT of magnitudes + random phase)
    # Build complex coefficients using magnitudes from fd and zero phase:
    mag = np.zeros(n_coeffs, dtype=float)
    mag[1:len(fd)+1] = fd  # place magnitudes
    # create symmetric spectrum for real signal
    spec = np.zeros(n_coeffs, dtype=complex)
    # put magnitudes with zero phase
    spec[:len(mag)] = mag * np.exp(1j * 0.0)
    recon_seq = np.fft.ifft(spec)
    # plot real part shape
    seq = recon_seq.real
    seq = seq - seq.mean()
    # create scatter of descriptor reconstruction
    xsr = np.real(seq)
    ysr = np.imag(seq) if np.iscomplexobj(seq) else np.zeros_like(seq)
    ax[1].plot(xsr, ysr, '-', linewidth=1)
    ax[1].set_title("Descriptor-based shape (approx.)")
    ax[1].axis('equal')
    ax[1].axis('off')
    st.pyplot(fig)

    # if user uploads another symbol for comparison
    st.markdown("**Compare with another symbol** (optional upload)")
    uploaded2 = st.file_uploader("Upload second symbol for comparison", type=['png','jpg','jpeg'], key="comp")
    if uploaded2:
        arr2, _ = load_image_to_array(uploaded2, size=size, invert=True)
        fd2 = compute_fourier_descriptor(arr2, n_points=n_coeffs)
        d = descriptor_distance(fd, fd2)
        corr = np.sum(np.abs(np.fft.fft2(arr)) * np.abs(np.fft.fft2(arr2)))
        st.write(f"Descriptor distance: {d:.4f}")
        st.write(f"FFT magnitude correlation (raw): {corr:.4g}")
        # show side-by-side
        figc, axc = plt.subplots(1,2, figsize=(6,3))
        axc[0].imshow(arr, cmap='gray'); axc[0].set_title("Symbol A"); axc[0].axis('off')
        axc[1].imshow(arr2, cmap='gray'); axc[1].set_title("Symbol B"); axc[1].axis('off')
        st.pyplot(figc)

st.markdown("---")
st.caption("Notes: These implementations are lightweight approximations intended for exploration. Gerchberg–Saxton runs on CPU and is not optimized for large arrays; adjust `size` down if it takes too long. WAV export maps the 2D field to audio by summing column carriers — creative but not physically rigorous. Fourier descriptors are boundary-based and made rotation/scale-invariant by centering and magnitude-based normalization.")
