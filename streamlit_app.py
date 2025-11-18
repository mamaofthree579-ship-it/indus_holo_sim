# streamlit_app.py
"""
Robust single-file Symbol Simulator for Indus Holo project.
- Self-contained (no fragile package-relative imports)
- Loads glyph registry if available
- Accepts uploads and NB lookups
- FFT / Fresnel / Acoustic / 2-beam holography modes
- Safe, type-consistent Streamlit widgets
"""
import streamlit as st
from pathlib import Path
import json, io, math
from PIL import Image, ImageOps
import numpy as np
import base64

st.set_page_config(layout="wide", page_title="Indus Holo — Simulator")

# ---------------------------
# Paths (try a few sensible locations)
# ---------------------------
ROOT = Path(__file__).resolve().parent
CANDIDATE_REGISTRIES = [
    Path("/tmp/nb_signs.json"),
    Path("/tmp/glyph_registry.json"),
    ROOT / "data" / "glyph_registry.json",
    ROOT / "data" / "nb_signs.json",
    ROOT / "data" / "mahadevan_registry.json",
]

def find_registry():
    for p in CANDIDATE_REGISTRIES:
        if p.exists():
            return p
    return None

# ---------------------------
# Registry loader: safe parse + normalization
# ---------------------------
def load_registry(path=None):
    path = path or find_registry()
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # normalize keys to NB### strings
        normalized = {}
        for k,v in data.items():
            key = str(k)
            if not key.upper().startswith("NB"):
                # try numeric
                try:
                    n = int(key)
                    key = f"NB{n:03d}"
                except:
                    pass
            normalized[key.upper()] = v or {}
        return normalized
    except Exception as e:
        st.warning(f"Failed loading registry {path}: {e}")
        return {}

# ---------------------------
# Glyph loading: many fallbacks
# ---------------------------
def load_glyph_by_nb(nb_id):
    nb_id = nb_id.upper()
    # possible local files
    paths_to_try = [
        Path("/tmp/nb_glyphs") / f"{nb_id}.png",
        ROOT / "data" / "glyphs" / "png" / f"{nb_id}.png",
        ROOT / "data" / "glyphs" / f"{nb_id}.png",
        ROOT / "data" / f"{nb_id}.png",
    ]
    for p in paths_to_try:
        if p.exists():
            try:
                return Image.open(p).convert("L")
            except Exception:
                pass
    # try registry mapping
    reg = load_registry()
    entry = reg.get(nb_id)
    if entry:
        # try mask_path then png_path then svg_path (svg not supported here)
        for key in ["png_path", "mask_path", "glyph_path", "svg_path"]:
            v = entry.get(key)
            if v:
                p = Path(v)
                if p.exists():
                    try:
                        return Image.open(p).convert("L")
                    except Exception:
                        pass
    return None

# ---------------------------
# Procedural fallback glyph generator (simple, but readable)
# ---------------------------
def procedural_glyph_image(nb_id, size=256):
    # deterministic pseudo-historical shape via seed
    from PIL import ImageDraw
    seed = abs(hash(nb_id)) % (2**32)
    rng = np.random.RandomState(seed % 4294967295)
    img = Image.new("L", (size,size), color=255)
    draw = ImageDraw.Draw(img)
    cx,cy = size//2, size//2
    nstrokes = 3 + (seed % 4)
    for i in range(nstrokes):
        angle = rng.rand() * 2*math.pi
        r = size* (0.12 + 0.08*i)
        x1 = cx + r*math.cos(angle)
        y1 = cy + r*math.sin(angle)
        x2 = cx + r*0.4*math.cos(angle+0.9)
        y2 = cy + r*0.4*math.sin(angle+0.9)
        draw.line([x1,y1,x2,y2], fill=0, width=6 - min(4,i))
    # invert so ink=black -> bright for our pipeline (we'll invert later as needed)
    return img

# ---------------------------
# Image to working numpy
# ---------------------------
def pil_to_norm_np(im, target_size=256, invert=True):
    """Return float array in 0..1 with 'ink' as 1.0"""
    im = im.copy()
    if invert:
        im = ImageOps.invert(im)
    im = ImageOps.autocontrast(im)
    im = im.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(im, dtype=float) / 255.0
    return arr

# ---------------------------
# Simulation primitives
# ---------------------------
def simulate_fft(image):
    # Fraunhofer (FFT) based diffraction mag
    F = np.fft.fftshift(np.fft.fft2(image))
    mag = np.abs(F)
    mag = mag ** 0.4
    mag /= (mag.max() + 1e-12)
    return mag

def simulate_fresnel_symbol_preserve(image, distance=0.8, wavelength=0.00065, lowpass=0.25, blend=0.45):
    img = image.astype(float)
    img /= (img.max() + 1e-12)
    N = img.shape[0]
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
    intensity /= (intensity.max() + 1e-12)
    blended = (1 - blend) * intensity + blend * img
    blended /= (blended.max() + 1e-12)
    return blended, intensity

def simulate_acoustic(image, base_freq=5.0, harmonics=6):
    h,w = image.shape
    y = np.linspace(0,1,h); x = np.linspace(0,1,w)
    xx,yy = np.meshgrid(x,y)
    field = np.zeros_like(image, dtype=float)
    for k in range(1, harmonics+1):
        field += image * np.sin(2*math.pi*(base_freq*k)*(xx+yy))
    field = np.abs(field)
    field /= (field.max()+1e-12)
    return field

def simulate_two_beam_hologram(image, wavelength=0.05, angle_deg=45, phase_deg=0, amplitude_ref=1.0):
    angle = math.radians(angle_deg)
    h,w = image.shape
    y = np.linspace(0,1,h); x = np.linspace(0,1,w)
    xx,yy = np.meshgrid(x,y)
    carrier = np.cos(2*math.pi*(xx*math.cos(angle) + yy*math.sin(angle))/wavelength + math.radians(phase_deg))
    holo = (image * carrier * amplitude_ref)
    holo = holo - holo.min()
    holo /= (holo.max()+1e-12)
    return holo

# ---------------------------
# UI: Layout
# ---------------------------
st.title("Indus Holo — Main Simulator")
st.write("Use this page to load NB glyphs or upload your own, then run light / acoustic / holographic tests.")

# Left column: glyph selection / registry
left, right = st.columns([1,2])

with left:
    st.header("Glyph selection")

    registry = load_registry()
    registry_path = find_registry()
    if registry_path:
        st.write("Loaded registry from:", registry_path)
    else:
        st.info("No registry found. You can upload glyphs or use NB codes / procedural fallback.")

    # selection options
    sel_mode = st.selectbox("Input mode", ["Upload image", "Choose NB code", "Procedural fallback"])

    uploaded_file = None
    selected_nb = None
    glyph_pil = None

    if sel_mode == "Upload image":
        uploaded_file = st.file_uploader("Upload glyph image (PNG/JPG)", type=["png","jpg","jpeg"])
        if uploaded_file:
            try:
                glyph_pil = Image.open(uploaded_file).convert("L")
            except Exception as e:
                st.error("Failed to open uploaded image: " + str(e))

    elif sel_mode == "Choose NB code":
        # choose from registry list or type code
        nb_list = sorted(list(registry.keys()))
        nb_choice = st.selectbox("Pick NB (from registry)", ["--none--"] + nb_list)
        nb_text = st.text_input("or type NB code (e.g. NB001) ", value="")
        if nb_choice and nb_choice != "--none--":
            selected_nb = nb_choice
        elif nb_text.strip():
            selected_nb = nb_text.strip().upper()
        if selected_nb:
            glyph_pil = load_glyph_by_nb(selected_nb)
            if glyph_pil is None:
                st.warning(f"No glyph file found for {selected_nb}; you can use procedural fallback or upload an image.")
    else:
        nb_text2 = st.text_input("Name procedural glyph as (NB### or label)", value="NB000")
        selected_nb = nb_text2.strip() or "NB000"
        glyph_pil = procedural_glyph_image(selected_nb, size=256)

    # preview & download glyph
    if glyph_pil is not None:
        st.subheader("Glyph preview")
        st.image(glyph_pil, width=220)
        buf = io.BytesIO()
        glyph_pil.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download glyph PNG", data=buf, file_name=f"{(selected_nb or 'uploaded')}.png", mime="image/png")

    st.markdown("---")
    st.write("If registry entries have metadata it will appear on the right after selection.")
    if selected_nb and registry:
        meta = registry.get(selected_nb, {})
        st.write("Metadata (registry):")
        st.json(meta)

with right:
    st.header("Simulation controls & outputs")

    if glyph_pil is None:
        st.warning("No glyph loaded. Pick or upload one on the left.")
        st.stop()

    # preprocessing options
    size = st.slider("Working size (px)", 64, 512, 256)
    invert = st.checkbox("Invert (make ink=bright)", value=True)
    img_np = pil_to_norm_np(glyph_pil, target_size=size, invert=invert)

    mode = st.radio("Simulation mode", ["Light FFT (Fraunhofer)", "Enhanced Fresnel", "Acoustic Harmonics", "2-beam Hologram", "Compare modes"])

    result_img = None

    if mode == "Light FFT (Fraunhofer)":
        st.subheader("Fraunhofer diffraction (FFT magnitude)")
        res = simulate_fft(img_np)
        result_img = res
        st.image(result_img, clamp=True, use_column_width=True)

    elif mode == "Enhanced Fresnel":
        st.subheader("Fresnel diffusion (symbol-preserving)")
        distance = st.slider("Distance factor", 0.1, 2.0, 0.8, step=0.1)
        wavelength = st.slider("Wavelength (arbitrary)", 0.0001, 0.0050, 0.00065, step=0.00005)
        lowpass = st.slider("Low-pass fraction", 0.05, 1.0, 0.25)
        blend = st.slider("Blend with symbol", 0.0, 1.0, 0.45)
        enhanced, raw = simulate_fresnel_symbol_preserve(img_np, distance=distance, wavelength=wavelength, lowpass=lowpass, blend=blend)
        st.write("Raw intensity (Fresnel)")
        st.image(raw, clamp=True, use_column_width=True)
        st.write("Symbol-preserving blended result")
        st.image(enhanced, clamp=True, use_column_width=True)
        result_img = enhanced

    elif mode == "Acoustic Harmonics":
        st.subheader("Acoustic harmonic response")
        base_freq = st.slider("Base frequency (arbitrary)", 1, 40, 5)
        harmonics = st.slider("Harmonics", 1, 20, 6)
        ac = simulate_acoustic(img_np, base_freq=base_freq, harmonics=harmonics)
        st.image(ac, clamp=True, use_column_width=True)
        result_img = ac

    elif mode == "2-beam Hologram":
        st.subheader("Two-beam holographic interference (carrier)")
        wavelength = st.slider("Wavelength (arb.)", 0.01, 0.2, 0.05)
        angle = st.slider("Angle (deg)", 0, 180, 45)
        phase = st.slider("Phase (deg)", 0, 360, 0)
        amplitude_ref = st.slider("Reference amplitude", 0.0, 2.0, 1.0)
        holo = simulate_two_beam_hologram(img_np, wavelength=wavelength, angle_deg=angle, phase_deg=phase, amplitude_ref=amplitude_ref)
        st.image(holo, clamp=True, use_column_width=True)
        result_img = holo

    elif mode == "Compare modes":
        st.subheader("Compare several modes side-by-side")
        e_d, e_wl, lp, blend = 0.8, 0.00065, 0.25, 0.45
        fra = simulate_fft(img_np)
        fres, raw = simulate_fresnel_symbol_preserve(img_np, distance=e_d, wavelength=e_wl, lowpass=lp, blend=blend)
        ac = simulate_acoustic(img_np, base_freq=5, harmonics=6)
        holo = simulate_two_beam_hologram(img_np, wavelength=0.05, angle_deg=45)
        cols = st.columns(4)
        with cols[0]:
            st.caption("FFT")
            st.image(fra, clamp=True, use_column_width=True)
        with cols[1]:
            st.caption("Fresnel")
            st.image(fres, clamp=True, use_column_width=True)
        with cols[2]:
            st.caption("Acoustic")
            st.image(ac, clamp=True, use_column_width=True)
        with cols[3]:
            st.caption("2-beam")
            st.image(holo, clamp=True, use_column_width=True)
        result_img = fres

    # download result as PNG
    if result_img is not None:
        out_pil = Image.fromarray((np.clip(result_img,0,1)*255).astype(np.uint8))
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download result PNG", data=buf, file_name=f"sim_result_{mode.replace(' ','_')}.png", mime="image/png")

st.write("---")
st.caption("If you want this app to call your other pages (Diffraction Lab, Advanced tester) I can add direct navigation buttons and link registry updates. This app is intentionally self-contained to avoid import/path issues.")
