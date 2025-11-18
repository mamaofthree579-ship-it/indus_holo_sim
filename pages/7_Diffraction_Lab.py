# pages/8_Diffraction_Physics_Advanced.py
"""
Diffraction Physics — Advanced toolbox
Features:
 - Phase+Amplitude hologram synthesis
 - Angular spectrum propagation (pseudo-3D volumetric holography)
 - Color (multi-wavelength) diffraction rendering
 - Inverse propagation & Gerchberg-Saxton reconstruction tools
 - Realistic noise models (speckle, surface scattering, blur)
Pure numpy + Pillow + matplotlib + streamlit.
"""
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import io, zipfile, math
import matplotlib.pyplot as plt


st.set_page_config(layout="wide", page_title="Diffraction Physics — Advanced")
st.title("Diffraction Physics — Advanced Toolkit")

# ---------------------------
# Utilities
# ---------------------------
def load_image_arr(uploaded, size=256, invert=True):
    im = Image.open(uploaded).convert("L")
    im = ImageOps.autocontrast(im)
    if invert:
        im = ImageOps.invert(im)
    im = im.resize((size, size), Image.LANCZOS)
    arr = np.array(im, dtype=float) / 255.0
    return arr, im

def arr_to_image(a, cmap='gray'):
    a = np.clip(a, 0, 1)
    im = Image.fromarray((a*255).astype(np.uint8))
    return im

def pil_from_array_rgb(arr):
    # arr float [H,W,3] 0..1
    arr = np.clip(arr*255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def download_bytes(data_bytes, name, mime):
    st.download_button(name, data_bytes, file_name=name, mime=mime)

# ---------------------------
# Angular Spectrum Propagation
# ---------------------------
def angular_spectrum_propagation(u0, dx, wavelength, z):
    """
    Angular spectrum method for propagation of field u0 (complex) a distance z.
    u0: 2D complex field (H x W)
    dx: pixel pitch (arbitrary units)
    wavelength: same units as dx
    z: propagation distance (same units)
    returns complex field at z
    """
    k = 2*np.pi / wavelength
    ny, nx = u0.shape
    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dx)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * z * 2*np.pi * np.sqrt(np.maximum(0.0, (1.0/wavelength**2) - (FX**2 + FY**2))))
    U0 = np.fft.fft2(u0)
    Uz = U0 * H
    uz = np.fft.ifft2(Uz)
    return uz

# ---------------------------
# Fresnel / Fraunhofer helpers (for convenience)
# ---------------------------
def fraunhofer_intensity(aperture):
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))
    I = np.abs(F)**2
    return I / (I.max() + 1e-12)

# ---------------------------
# Phase+Amplitude hologram synthesis
# ---------------------------
def synthesize_phase_amplitude(target_amp, aperture_mask=None, n_iters=50):
    """
    Create a complex hologram H such that |FFT(H)| ~ target_amp
    Returns complex hologram (amplitude*exp(i*phase)).
    This is a Gerchberg-like loop but we keep amplitude in aperture.
    aperture_mask: optional binary mask for aperture support in pupil plane.
    """
    M, N = target_amp.shape
    rng = np.random.default_rng(42)
    # initial complex hologram (random phase, amplitude = aperture_mask or 1)
    if aperture_mask is None:
        aperture = np.ones((M,N))
    else:
        aperture = aperture_mask.astype(float)
    H = aperture * np.exp(1j * 2*np.pi * rng.random((M,N)))

    for _ in range(n_iters):
        # forward
        F = np.fft.fft2(H)
        phaseF = np.angle(F)
        # replace amplitude with target_amp
        F_new = target_amp * np.exp(1j*phaseF)
        # backpropagate
        H_new = np.fft.ifft2(F_new)
        # enforce aperture amplitude constraints
        H = aperture * (np.abs(H_new) / (np.abs(H_new) + 1e-12)) * np.exp(1j * np.angle(H_new))
    return H

# ---------------------------
# Color diffraction: simulate for multiple wavelengths and combine
# ---------------------------
def color_diffraction(aperture_float, dx, wavelengths, z, method='fraunhofer'):
    """
    aperture_float: 2D amplitude map (0..1)
    wavelengths: list of three wavelengths for R,G,B (units consistent with dx)
    z: propagation distance for angular spectrum (if used)
    method: 'fraunhofer' or 'angular'
    returns RGB float image 0..1
    """
    H,W = aperture_float.shape
    rgb = np.zeros((H,W,3), dtype=float)
    for i, wl in enumerate(wavelengths):
        if method == 'fraunhofer':
            I = fraunhofer_intensity(aperture_float * np.exp(0j))
            rgb[:,:,i] = I
        else:
            # angular spectrum: aperture as complex amplitude (real amplitude)
            u0 = aperture_float.astype(np.complex128)
            uz = angular_spectrum_propagation(u0, dx, wl, z)
            I = np.abs(uz)**2
            rgb[:,:,i] = I / (I.max() + 1e-12)
    # normalize composite so overall max =1
    rgb = rgb / (rgb.max() + 1e-12)
    return rgb

# ---------------------------
# Inverse propagation & GS recovery helper
# ---------------------------
def gerchberg_saxon_recovery(measured_intensity, aperture_mask=None, n_iters=150):
    """
    Try to recover an aperture field whose far-field magnitude is measured_intensity.
    measured_intensity: desired far-field amplitude (not intensity) -> we accept sqrt(intensity)
    Returns complex aperture field estimate.
    """
    target_amp = np.sqrt(np.maximum(measured_intensity, 0.0))
    M,N = target_amp.shape
    rng = np.random.default_rng(123)
    # initial aperture (random phase)
    if aperture_mask is None:
        aperture = np.ones((M,N))
    else:
        aperture = aperture_mask.copy()
    field = aperture * np.exp(1j * 2*np.pi * rng.random((M,N)))

    for k in range(n_iters):
        F = np.fft.fft2(field)
        phaseF = np.angle(F)
        F_new = target_amp * np.exp(1j*phaseF)
        field_new = np.fft.ifft2(F_new)
        # enforce aperture support amplitude
        field = aperture * np.exp(1j * np.angle(field_new))
    return field

# ---------------------------
# Noise models
# ---------------------------
def add_speckle(image, strength=0.3, grain=8):
    """
    Speckle: multiplicative grain noise. Strength 0..1.
    grain: size of coarse noise grain.
    """
    H,W = image.shape
    # generate coarse noise then smooth
    rng = np.random.default_rng(43)
    noise = rng.normal(loc=1.0, scale=strength, size=(H//grain+2, W//grain+2))
    # upscale via repeat
    noise = np.kron(noise, np.ones((grain, grain)))
    noise = noise[:H, :W]
    out = image * noise
    out = out / (out.max()+1e-12)
    return out

def add_surface_scatter(image, roughness=0.1):
    """
    Simulate scattering by convolving image with a small kernel dependent on roughness.
    """
    radius = max(1, int(roughness * 10))
    pil = Image.fromarray((image*255).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    arr = np.array(pil).astype(float)/255.0
    return arr

def add_contrast_fade(image, factor=0.7):
    out = image * factor + (1-factor)*0.5
    return (out - out.min())/(out.max()-out.min()+1e-12)

# ----- NAVIGATION -----
st.sidebar.markdown("## 🔀 Navigation")
st.sidebar.page_link("streamlit_app.py", label="🔮 Main Simulator")
st.sidebar.page_link("pages/0_Simulator_Admin.py", label="⚙️ Admin / Normalization")
st.sidebar.page_link("pages/7_Diffraction_Lab.py", label="🌈 Diffraction Physics Lab")
# -----------------------

# ---------------------------
# UI: Inputs
# ---------------------------
st.sidebar.header("Inputs & Options")
uploads = st.sidebar.file_uploader("Upload 1 symbol (PNG/JPG)", type=['png','jpg','jpeg'])
size = st.sidebar.slider("Processing size (px)", 64, 512, 256)
dx = st.sidebar.number_input("Pixel pitch (arbitrary units)", min_value=1e-6, value=1e-3, format="%.6f")
method_choice = st.sidebar.selectbox("Primary simulation mode", [
    "Phase+Amplitude Hologram",
    "Angular Spectrum Volume (z-stack)",
    "Color Diffraction (RGB wavelengths)",
    "Inverse Reconstruction (GS Recovery)",
    "Noise Robustness Tests"
])

if not uploads:
    st.info("Upload a cropped glyph image in the sidebar to proceed.")
    st.stop()

# preprocess
ap, pil_in = load_image_arr(uploads, size=size, invert=True)
st.subheader("Input Symbol (processed)")
st.image(pil_in, use_column_width=False)

# option: aperture mask (binary)
aperture_thresh = st.sidebar.slider("Aperture threshold for mask (0..255)", 1, 254, 20)
aperture_mask = (np.array(pil_in.convert("L")) < aperture_thresh).astype(float)
aperture_mask = Image.fromarray((aperture_mask*255).astype(np.uint8)).resize((size,size))
ap_mask = np.array(aperture_mask).astype(float)/255.0

# noise controls
st.sidebar.markdown("---")
st.sidebar.subheader("Noise models")
use_speckle = st.sidebar.checkbox("Apply speckle (multiplicative)", value=False)
speckle_strength = st.sidebar.slider("Speckle strength", 0.0, 1.0, 0.25)
use_scatter = st.sidebar.checkbox("Apply surface scatter (blur)", value=False)
scatter_rough = st.sidebar.slider("Surface roughness", 0.0, 1.0, 0.1)
use_contrast = st.sidebar.checkbox("Apply contrast fade", value=False)
contrast_factor = st.sidebar.slider("Contrast factor", 0.1, 1.0, 0.85)

# helper to apply noise chain
def apply_noise_chain(img):
    out = img.copy()
    if use_speckle: out = add_speckle(out, strength=speckle_strength, grain=max(2, int(size/64)))
    if use_scatter: out = add_surface_scatter(out, roughness=scatter_rough)
    if use_contrast: out = add_contrast_fade(out, factor=contrast_factor)
    return out

# ---------------------------
# Mode: Phase+Amplitude Hologram
# ---------------------------
if method_choice == "Phase+Amplitude Hologram":
    st.header("Phase + Amplitude Hologram Synthesis")
    n_iters = st.slider("GS-like iterations", 10, 400, 80)
    show_amp = st.checkbox("Show amplitude (aperture) and phase", value=True)
    # target amplitude: use FFT magnitude of input (far-field) or direct symbol
    target_choice = st.selectbox("Target amplitude", ["FFT magnitude (far-field)", "Symbol image (direct)"])
    if target_choice.startswith("FFT"):
        target = np.sqrt(fraunhofer_intensity(ap))
        target = target / (target.max()+1e-12)
    else:
        target = ap.copy()
    st.write("Synthesizing complex hologram...")
    H = synthesize_phase_amplitude(target, aperture_mask=ap_mask, n_iters=n_iters)
    amp = np.abs(H)
    phase = np.angle(H)
    # optionally apply noise then compute reconstruction
    amp_noisy = apply_noise_chain(amp)
    H_noisy = amp_noisy * np.exp(1j*phase)
    recon = fraunhofer_intensity(H_noisy)
    # display
    fig, axs = plt.subplots(1,4, figsize=(16,4))
    axs[0].imshow(target, cmap='inferno'); axs[0].set_title("Target amplitude"); axs[0].axis('off')
    axs[1].imshow(amp, cmap='gray'); axs[1].set_title("Hologram amplitude"); axs[1].axis('off')
    axs[2].imshow((phase+np.pi)/(2*np.pi), cmap='twilight'); axs[2].set_title("Hologram phase (0..1)"); axs[2].axis('off')
    axs[3].imshow(recon, cmap='magma'); axs[3].set_title("Reconstruction (far-field)"); axs[3].axis('off')
    st.pyplot(fig)
    # downloads
    buf_amp = io.BytesIO(); Image.fromarray((amp/amp.max()*255).astype(np.uint8)).save(buf_amp, format='PNG'); buf_amp.seek(0)
    buf_phase = io.BytesIO(); Image.fromarray(((phase+np.pi)/(2*np.pi)*255).astype(np.uint8)).save(buf_phase, format='PNG'); buf_phase.seek(0)
    st.download_button("Download amplitude PNG", buf_amp.getvalue(), file_name="hologram_amplitude.png", mime="image/png")
    st.download_button("Download phase PNG", buf_phase.getvalue(), file_name="hologram_phase.png", mime="image/png")
    # bundle both as zip
    zipb = io.BytesIO()
    with zipfile.ZipFile(zipb, "w") as z:
        z.writestr("hologram_amplitude.png", buf_amp.getvalue())
        z.writestr("hologram_phase.png", buf_phase.getvalue())
    zipb.seek(0)
    st.download_button("Download amplitude+phase ZIP", zipb.read(), file_name="hologram_bundle.zip", mime="application/zip")

# ---------------------------
# Mode: Angular Spectrum Volume
# ---------------------------
elif method_choice == "Angular Spectrum Volume (z-stack)":
    st.header("Angular Spectrum Volume (z-stack propagation)")
    z_min = st.number_input("z start", value=1e-3, format="%.6f")
    z_max = st.number_input("z end", value=1e-2, format="%.6f")
    z_steps = st.slider("z steps", 3, 64, 12)
    wavelength = st.number_input("Wavelength (same units as pixel pitch)", value=6.5e-7, format="%.8f")
    apply_noise = st.checkbox("Apply noise to aperture before propagation", value=False)
    # initial complex aperture field (amplitude=ap, phase=0)
    aperture = ap.copy()
    if apply_noise:
        aperture = apply_noise_chain(aperture)
    u0 = aperture.astype(np.complex128)
    zs = np.linspace(z_min, z_max, z_steps)
    gallery = []
    for z in zs:
        uz = angular_spectrum_propagation(u0, dx, wavelength, z)
        I = np.abs(uz)**2
        I = I / (I.max()+1e-12)
        gallery.append(I)
    # show 3-slice preview (start,middle,end)
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(gallery[0], cmap='magma'); axs[0].set_title(f"z={zs[0]:.3e}"); axs[0].axis('off')
    axs[1].imshow(gallery[len(gallery)//2], cmap='magma'); axs[1].set_title(f"z={zs[len(gallery)//2]:.3e}"); axs[1].axis('off')
    axs[2].imshow(gallery[-1], cmap='magma'); axs[2].set_title(f"z={zs[-1]:.3e}"); axs[2].axis('off')
    st.pyplot(fig)
    # pack stack into zip as PNGs
    bufz = io.BytesIO()
    with zipfile.ZipFile(bufz, "w") as zf:
        for i, I in enumerate(gallery):
            im = Image.fromarray((I/I.max()*255).astype(np.uint8))
            b = io.BytesIO(); im.save(b, format='PNG'); b.seek(0)
            zf.writestr(f"z_{i+1:03d}.png", b.getvalue())
    bufz.seek(0)
    st.download_button("Download z-stack PNGs (ZIP)", bufz.read(), file_name="zstack.zip", mime="application/zip")

# ---------------------------
# Mode: Color Diffraction
# ---------------------------
elif method_choice == "Color Diffraction (RGB wavelengths)":
    st.header("Color Diffraction (RGB wavelengths)")
    # default wavelengths in meters for R,G,B approx
    rwl = st.number_input("Red wavelength (m)", value=650e-9, format="%.9f")
    gwl = st.number_input("Green wavelength (m)", value=532e-9, format="%.9f")
    bwl = st.number_input("Blue wavelength (m)", value=451e-9, format="%.9f")
    z = st.number_input("Propagation distance (for angular)", value=1e-3, format="%.6f")
    method = st.selectbox("Method", ["fraunhofer", "angular"])
    ap_for_color = ap.copy()
    if use_speckle:
        ap_for_color = add_speckle(ap_for_color, speckle_strength)
    rgb = color_diffraction(ap_for_color, dx, [rwl, gwl, bwl], z, method=method)
    img_rgb = pil_from_array_rgb(rgb)
    st.image(img_rgb, caption="Simulated color diffraction", use_column_width=True)
    # download
    buf = io.BytesIO(); img_rgb.save(buf, format='PNG'); buf.seek(0)
    st.download_button("Download color diffraction (PNG)", buf.getvalue(), file_name="color_diffraction.png", mime="image/png")

# ---------------------------
# Mode: Inverse Reconstruction & GS Recovery
# ---------------------------
elif method_choice == "Inverse Reconstruction (GS Recovery)":
    st.header("Inverse Reconstruction & Gerchberg–Saxton Recovery")
    # simulate measurement (we'll measure far-field intensity of current aperture)
    measured_int = fraunhofer_intensity(ap)
    st.subheader("Simulated measured far-field intensity")
    plt.figure(figsize=(4,4))
    plt.imshow(measured_int, cmap='inferno'); plt.axis('off')
    st.pyplot(plt.gcf()); plt.clf()
    n_iter = st.slider("GS recovery iterations", 10, 500, 200)
    st.write("Running GS recovery (aperture support used)...")
    recovered = gerchberg_saxon_recovery(measured_int, aperture_mask=ap_mask, n_iters=n_iter)
    rec_amp = np.abs(recovered)
    rec_phase = np.angle(recovered)
    # display recovered aperture amplitude and phase and forward recon
    recon_forward = fraunhofer_intensity(recovered)
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(rec_amp, cmap='gray'); axs[0].set_title("Recovered aperture amplitude"); axs[0].axis('off')
    axs[1].imshow((rec_phase+np.pi)/(2*np.pi), cmap='twilight'); axs[1].set_title("Recovered phase (0..1)"); axs[1].axis('off')
    axs[2].imshow(recon_forward, cmap='magma'); axs[2].set_title("Forward reconstruction (from recovered)"); axs[2].axis('off')
    st.pyplot(fig)
    # download recovered amplitude/phase
    bufA = io.BytesIO(); Image.fromarray((rec_amp/rec_amp.max()*255).astype(np.uint8)).save(bufA, format='PNG'); bufA.seek(0)
    bufP = io.BytesIO(); Image.fromarray(((rec_phase+np.pi)/(2*np.pi)*255).astype(np.uint8)).save(bufP, format='PNG'); bufP.seek(0)
    st.download_button("Download recovered amplitude", bufA.getvalue(), "recovered_amplitude.png", mime="image/png")
    st.download_button("Download recovered phase", bufP.getvalue(), "recovered_phase.png", mime="image/png")

# ---------------------------
# Mode: Noise Robustness Tests
# ---------------------------
elif method_choice == "Noise Robustness Tests":
    st.header("Noise Robustness: Compare clean vs noisy diffraction")
    wl = st.number_input("Wavelength (m)", value=650e-9, format="%.9f")
    z = st.number_input("Propagation distance (m)", value=1e-3, format="%.6f")
    # clean diffraction
    clean = fraunhofer_intensity(ap)
    noisy_ap = apply_noise_chain(ap)
    noisy = fraunhofer_intensity(noisy_ap)
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(clean, cmap='magma'); axs[0].set_title("Clean aperture -> diffraction"); axs[0].axis('off')
    axs[1].imshow(noisy, cmap='magma'); axs[1].set_title("Noisy aperture -> diffraction"); axs[1].axis('off')
    st.pyplot(fig)
    st.write("You can tweak speckle/blur/contrast sliders in sidebar and re-run to see effects.")
    # provide side-by-side download
    b1 = io.BytesIO(); Image.fromarray((clean/clean.max()*255).astype(np.uint8)).save(b1, format='PNG'); b1.seek(0)
    b2 = io.BytesIO(); Image.fromarray((noisy/noisy.max()*255).astype(np.uint8)).save(b2, format='PNG'); b2.seek(0)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("clean_diffraction.png", b1.getvalue())
        zf.writestr("noisy_diffraction.png", b2.getvalue())
    zbuf.seek(0)
    st.download_button("Download compare ZIP", zbuf.read(), file_name="diffraction_noise_compare.zip", mime="application/zip")

st.write("---")
st.caption("Notes: units are arbitrary — keep pixel pitch and wavelength in same units when using angular propagation. Angular spectrum uses a simple square-root phase factor and omits evanescent component handling for simplicity. These tools are designed for experimentation and visualization; if you want physically calibrated units (meters, distances) we can add real-world scaling and sample spacing details next.")
