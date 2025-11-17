import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import io, zipfile
from collections import deque

st.title("Auto-Crop Indus Glyph Candidates (Updated Version)")

# -----------------------------------------
# 1. Locate all page image directories
# -----------------------------------------
candidate_dirs = [
    Path("/tmp/mahadevan_pages"),            # scraper output
    Path("/tmp/mahadevan_extracted/pages"),  # fallback
    Path("/tmp/mahadevan_extracted"),        # fallback
    Path("data/mahadevan_pages"),            # local storage
    Path("data/mahadevan_extracted/pages"),  # local fallback
]

existing_dirs = [d for d in candidate_dirs if d.exists()]

if not existing_dirs:
    st.error("""
    No page images found.  
    Please run **Page Scraper** first (pages/1_Scrape_Mahadevan_Images.py).
    """)
    st.stop()

# Pick best folder with images
PAGES = None
for d in existing_dirs:
    images = list(d.glob("*.jpg")) + list(d.glob("*.png"))
    if images:
        PAGES = d
        break

if PAGES is None:
    st.error("No JPG/PNG files found in detected directories.")
    st.stop()

st.success(f"Found page images in: {PAGES}")

# -----------------------------------------
# 2. Load pages
# -----------------------------------------
page_files = sorted(list(PAGES.glob("*.jpg")) + list(PAGES.glob("*.png")))

if not page_files:
    st.error("Pages folder exists but contains no images.")
    st.stop()

page_choice = st.selectbox("Choose a page image:", [p.name for p in page_files])
page_path = PAGES / page_choice

# -----------------------------------------
# 3. Cropping settings
# -----------------------------------------
threshold = st.slider("Binarization Threshold", 5, 250, 90)
min_area = st.number_input("Minimum Blob Area", min_value=20, value=250, step=10)
padding  = st.number_input("Crop Padding", min_value=0, value=5, step=1)

OUTPUT_DIR = Path("/tmp/glyph_candidates")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# 4. Crop Execution
# -----------------------------------------
if st.button("Detect & Crop Glyph Regions"):
    im = Image.open(page_path).convert("L")  # grayscale
    w, h = im.size

    # Preprocess
    im2 = ImageOps.autocontrast(im)
    im2 = im2.filter(ImageFilter.MedianFilter(size=3))
    bw = im2.point(lambda p: 255 if p < threshold else 0)

    arr = np.array(bw)
    visited = np.zeros(arr.shape, dtype=bool)
    H, W = arr.shape
    components = []

    # Connected Components
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

    components = sorted(components, key=lambda c: (c[1], c[0]))

    # Save crops
    saved = []
    for i, (mx, my, Mx, My, area) in enumerate(components):
        x0 = max(mx - padding, 0)
        y0 = max(my - padding, 0)
        x1 = min(Mx + padding, W)
        y1 = min(My + padding, H)

        crop = im.crop((x0, y0, x1, y1))
        outpath = OUTPUT_DIR / f"{page_choice}_crop_{i+1:03d}.png"
        crop.save(outpath)
        saved.append(outpath)

    st.success(f"Extracted {len(saved)} glyph candidates.")

    cols = st.columns(3)
    for i, s in enumerate(saved[:12]):
        with cols[i % 3]:
            st.image(str(s), caption=s.name)

    # Make ZIP
    if saved:
        zpath = Path("/tmp") / f"{page_choice}_glyphs.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            for s in saved:
                z.write(s, s.name)
        with open(zpath, "rb") as f:
            st.download_button("Download Cropped Glyph ZIP", f, file_name=zpath.name)
