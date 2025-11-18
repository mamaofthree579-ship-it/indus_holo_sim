import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from collections import deque
import io, zipfile

st.title("Upload Page → Auto-Crop Indus Glyphs (Upload-Based Version)")

# -------------------------------------------------
# 1. Upload page image
# -------------------------------------------------
uploaded = st.file_uploader("Upload a Mahadevan page image (JPG/PNG)", 
                            type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    st.image(img, caption="Uploaded Page", use_column_width=True)

    # -------------------------------------------------
    # 2. Cropping controls
    # -------------------------------------------------
    threshold = st.slider("Binarization Threshold", 5, 250, 90)
    min_area  = st.number_input("Minimum Glyph Blob Area", 
                                min_value=20, value=250, step=10)
    padding   = st.number_input("Crop Padding (px)", 
                                min_value=0, value=5, step=1)

    # -------------------------------------------------
    # 3. Auto-crop button
    # -------------------------------------------------
    if st.button("Detect & Crop Glyphs"):

        w, h = img.size

        # Preprocessing
        im2 = ImageOps.autocontrast(img)
        im2 = im2.filter(ImageFilter.MedianFilter(size=3))
        bw = im2.point(lambda p: 255 if p < threshold else 0)

        arr = np.array(bw)
        visited = np.zeros(arr.shape, dtype=bool)
        H, W = arr.shape
        components = []

        # -------------------------------------------------
        # Connected Component Detection
        # -------------------------------------------------
        for y in range(H):
            for x in range(W):
                if arr[y, x] == 255 and not visited[y, x]:
                    q = deque()
                    q.append((y, x))
                    visited[y, x] = True
                    ys = [y]; xs = [x]

                    while q:
                        yy, xx = q.popleft()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = yy + dy, xx + dx
                                if 0 <= ny < H and 0 <= nx < W:
                                    if not visited[ny, nx] and arr[ny, nx] == 255:
                                        visited[ny, nx] = True
                                        q.append((ny, nx))
                                        ys.append(ny); xs.append(nx)

                    minx, maxx = min(xs), max(xs)
                    miny, maxy = min(ys), max(ys)
                    area = (maxx - minx) * (maxy - miny)

                    if area >= min_area:
                        components.append((minx, miny, maxx, maxy, area))

        # Sort glyphs row-by-row
        components = sorted(components, key=lambda c: (c[1], c[0]))

        # -------------------------------------------------
        # Crop and collect glyphs
        # -------------------------------------------------
        cropped_images = []
        for i, (mx, my, Mx, My, area) in enumerate(components):
            x0 = max(mx - padding, 0)
            y0 = max(my - padding, 0)
            x1 = min(Mx + padding, W)
            y1 = min(My + padding, H)

            crop = img.crop((x0, y0, x1, y1))

            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            buf.seek(0)

            cropped_images.append((f"glyph_{i+1:03d}.png", buf))

        st.success(f"Detected {len(cropped_images)} glyph candidates.")

        # -------------------------------------------------
        # Display top examples
        # -------------------------------------------------
        cols = st.columns(3)
        for i, (name, buf) in enumerate(cropped_images[:12]):
            with cols[i % 3]:
                st.image(buf, caption=name, use_column_width=True)

        # -------------------------------------------------
        # Download ZIP
        # -------------------------------------------------
        if cropped_images:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as z:
                for name, buf in cropped_images:
                    z.writestr(name, buf.getvalue())
            zip_buf.seek(0)

            st.download_button(
                "Download All Cropped Glyphs (ZIP)",
                zip_buf,
                "cropped_glyphs.zip",
                mime="application/zip"
            )
