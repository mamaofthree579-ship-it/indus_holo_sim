# pages/2_Auto_Crop_Glyphs.py
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os, io
import zipfile

st.title("Auto-crop glyph candidates from Mahadevan pages")

PAGES_DIR = Path("data") / "mahadevan_extracted" / "pages_png"
OUT_GLYPH_DIR = Path("data") / "mahadevan_glyphs_auto"
OUT_GLYPH_DIR.mkdir(parents=True, exist_ok=True)

page_files = sorted(PAGES_DIR.glob("*.png"))
if not page_files:
    st.error("No page PNGs found. Run the extraction page first.")
    st.stop()

page_choice = st.selectbox("Choose page to crop", [p.name for p in page_files])
threshold = st.slider("Binarization threshold", 10, 240, 80)
min_area = st.number_input("Min component area (px)", min_value=20, max_value=100000, value=200, step=10)
pad = st.number_input("Crop padding (px)", min_value=0, max_value=100, value=6, step=1)

if st.button("Auto-crop this page"):
    pth = PAGES_DIR / page_choice
    im = Image.open(pth).convert("L")
    w,h = im.size
    # basic contrast increase + blur to help
    im2 = ImageOps.autocontrast(im)
    im2 = im2.filter(ImageFilter.MedianFilter(size=3))
    bw = im2.point(lambda p: 255 if p < threshold else 0)  # invert: ink=white
    bw = bw.convert("L")
    arr = np.array(bw)
    # find connected components (simple flood fill)
    visited = np.zeros(arr.shape, dtype=bool)
    H,W = arr.shape
    components = []
    def neighbors(y,x):
        for ny in (y-1,y,y+1):
            for nx in (x-1,x,nx := x+1):
                pass
    # efficient component detection using scipy would be ideal, but do a quick scan:
    from collections import deque
    for y in range(H):
        for x in range(W):
            if arr[y,x] == 255 and not visited[y,x]:
                q = deque()
                q.append((y,x))
                visited[y,x] = True
                ys=[y]; xs=[x]
                while q:
                    yy,xx = q.popleft()
                    for dy in (-1,0,1):
                        for dx in (-1,0,1):
                            ny, nx = yy+dy, xx+dx
                            if 0 <= ny < H and 0 <= nx < W and not visited[ny,nx] and arr[ny,nx]==255:
                                visited[ny,nx]=True
                                q.append((ny,nx))
                                ys.append(ny); xs.append(nx)
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                area = (maxx-minx)*(maxy-miny)
                if area >= min_area:
                    components.append((minx,miny,maxx,maxy,area))
    # sort components left-to-right, top-to-bottom
    components = sorted(components, key=lambda r: (r[1], r[0]))
    # save crops
    saved = []
    for idx, (minx,miny,maxx,maxy,area) in enumerate(components):
        mx0 = max(minx-pad,0); my0 = max(miny-pad,0)
        mx1 = min(maxx+pad,W); my1 = min(maxy+pad,H)
        crop = im.crop((mx0,my0,mx1,my1))
        outp = OUT_GLYPH_DIR / f"{page_choice}_cand_{idx+1:03d}.png"
        crop.save(outp)
        saved.append(str(outp))
    st.success(f"Saved {len(saved)} glyph candidates to {OUT_GLYPH_DIR}")
    for s in saved[:12]:
        st.image(s, width=120)
    if saved:
        # create zip
        zip_path = OUT_GLYPH_DIR / f"{page_choice}_candidates.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            for s in saved:
                z.write(s, Path(s).name)
        with open(zip_path, "rb") as f:
            st.download_button("Download this page's candidates (zip)", f.read(), file_name=zip_path.name)
