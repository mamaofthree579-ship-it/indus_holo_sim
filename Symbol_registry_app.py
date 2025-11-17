import streamlit as st
from pathlib import Path
import json, requests, io
from PIL import Image, ImageDraw
import hashlib, math
import xml.etree.ElementTree as ET
import zipfile
import base64

# ------------------------------------------------------------
# Setup directories relative to THIS file (works in Streamlit)
# ------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent  # go up to main project
DATA = BASE / "data"
SVG_DIR = DATA / "glyphs/svg"
PNG_DIR = DATA / "glyphs/png"
MASK_DIR = DATA / "glyphs/masks"

for d in [SVG_DIR, PNG_DIR, MASK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REGISTRY_PATH = DATA / "glyph_registry.json"

st.title("Indus Glyph Generator (Historical + Procedural)")
st.write("Generate and download all NB glyphs directly from Streamlit.")

# ------------------------------
# NB → M placeholder
# ------------------------------
def nb_to_m(nb):
    return nb  # adjust when mapping exists


# ------------------------------
# Try downloading SVG
# ------------------------------
def try_download_svg(m_num):
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/Indus_script_sign_{m_num:03d}.svg"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.text.startswith("<svg"):
            path = SVG_DIR / f"{m_num:03d}.svg"
            path.write_text(r.text, encoding="utf-8")
            return path
    except:
        return None
    return None


# ------------------------------
# Simple SVG → mask
# ------------------------------
def svg_to_mask(svg_path, size=512):
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except:
        return None

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    for elem in root.iter():
        if "path" in elem.tag:
            d = elem.attrib.get("d", "")
            pts = []
            tokens = d.replace(",", " ").split()

            i = 0
            while i < len(tokens):
                if tokens[i] in ("M", "L"):
                    try:
                        x = float(tokens[i+1])
                        y = float(tokens[i+2])
                        pts.append((x, y))
                    except:
                        pass
                    i += 3
                else:
                    i += 1
            if len(pts) > 1:
                draw.line(pts, fill=255, width=20)

    return mask


# ------------------------------
# Procedural fallback
# ------------------------------
def procedural_glyph(nb_code, size=512):
    seed = int(hashlib.sha256(nb_code.encode()).hexdigest(), 16)
    img = Image.new("RGBA", (size, size), (255,255,255,0))
    draw = ImageDraw.Draw(img)
    cx, cy = size//2, size//2

    for i in range(4 + seed % 4):
        ang = (seed >> (i*3)) & 0xFF
        th = ang / 255 * 2 * math.pi
        r1 = size * (0.15 + i*0.1)
        x1 = cx + r1 * math.cos(th)
        y1 = cy + r1 * math.sin(th)
        x2 = cx + (r1*0.4) * math.cos(th + 0.7)
        y2 = cy + (r1*0.4) * math.sin(th + 0.7)
        draw.line([x1,y1,x2,y2], fill=(0,0,0,255), width=10)

    mask = img.convert("L").point(lambda p: 255 if p > 20 else 0)
    return img, mask


# ------------------------------
# Build registry from Streamlit
# ------------------------------
def build_registry():
    registry = {}

    progress = st.progress(0)
    total = 417

    for i, nb in enumerate(range(1, total + 1)):
        nb_code = f"NB{nb:03d}"

        # Update progress bar
        progress.progress((i + 1) / total)
        st.write(f"Processing {nb_code}")

        m_num = nb_to_m(nb)
        svg_path = try_download_svg(m_num)

        if svg_path:
            mask = svg_to_mask(svg_path)
            if mask:
                png_path = PNG_DIR / f"{nb_code}.png"
                mask_path = MASK_DIR / f"{nb_code}_mask.png"

                mask_img = mask.convert("RGBA")
                mask_img.save(png_path)
                mask.save(mask_path)

                registry[nb_code] = {
                    "glyph_type": "historical",
                    "svg_path": str(svg_path),
                    "png_path": str(png_path),
                    "mask_path": str(mask_path)
                }
                continue

        # Procedural fallback
        img, mask = procedural_glyph(nb_code)
        png_path = PNG_DIR / f"{nb_code}.png"
        mask_path = MASK_DIR / f"{nb_code}_mask.png"
        img.save(png_path)
        mask.save(mask_path)

        registry[nb_code] = {
            "glyph_type": "procedural",
            "svg_path": None,
            "png_path": str(png_path),
            "mask_path": str(mask_path)
        }

    # Save registry
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    st.success("Registry created!")


# ------------------------------------
# UI: Button to generate everything
# ------------------------------------
if st.button("Generate All Glyphs (NB001–NB417)"):
    build_registry()


# ------------------------------------
# UI: Once registry exists—preview + download
# ------------------------------------
if REGISTRY_PATH.exists():
    st.subheader("Preview & Download")

    registry = json.loads(REGISTRY_PATH.read_text())

    nb_choice = st.selectbox("Choose a sign", sorted(registry.keys()))

    entry = registry[nb_choice]

    img = Image.open(entry["png_path"])
    st.image(img, caption=f"{nb_choice} ({entry['glyph_type']})", width=300)

    # Download individual PNG
    with open(entry["png_path"], "rb") as f:
        st.download_button(
            "Download PNG",
            data=f.read(),
            file_name=f"{nb_choice}.png",
            mime="image/png"
        )

    # ZIP download of all PNGs
    if st.button("Download ALL PNGs as ZIP"):
        zip_path = DATA / "all_glyphs.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            for nb, ent in registry.items():
                z.write(ent["png_path"], f"{nb}.png")
        with open(zip_path, "rb") as f:
            st.download_button(
                "Download ZIP",
                data=f.read(),
                file_name="IndusGlyphs.zip",
                mime="application/zip"
            )
