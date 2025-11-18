import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io, zipfile

def load_img(file, size=256):
    im = Image.open(file).convert("L")
    im = ImageOps.autocontrast(im)
    im = ImageOps.invert(im)
    im = im.resize((size, size))
    return np.array(im, float)/255.0

def gs(target, iters=150):
    M,N = target.shape
    rng = np.random.default_rng(1234)
    A = np.exp(1j * 2*np.pi * rng.random((M,N)))
    for _ in range(iters):
        F = np.fft.fft2(A)
        A = np.fft.ifft2(target * np.exp(1j * np.angle(F)))
        A = np.exp(1j * np.angle(A))
    return np.mod(np.angle(A) + 2*np.pi, 2*np.pi)

st.title("Batch Gerchberg–Saxton Phase Hologram Exporter")

uploads = st.file_uploader("Upload multiple symbols", 
                           accept_multiple_files=True,
                           type=["png","jpg","jpeg"])

size = st.sidebar.slider("Resize to (px)", 64, 512, 256)
iters = st.sidebar.slider("Iterations", 20, 500, 150)

if uploads:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        for f in uploads:
            arr = load_img(f, size=size)
            phase = gs(arr, iters)
            
            out = (phase/(2*np.pi)*255).astype(np.uint8)
            img = Image.fromarray(out)
            
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            z.writestr(f"{f.name}_phase.png", buf.read())

    zip_buf.seek(0)
    st.download_button("Download ZIP of All Phase Maps",
                       data=zip_buf,
                       file_name="phase_maps.zip",
                       mime="application/zip")
