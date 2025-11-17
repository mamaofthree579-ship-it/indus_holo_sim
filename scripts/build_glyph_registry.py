import json, requests
from pathlib import Path
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import hashlib, math

REGISTRY_PATH = Path("data/glyph_registry.json")
SVG_DIR = Path("data/glyphs/svg")
PNG_DIR = Path("data/glyphs/png")
MASK_DIR = Path("data/glyphs/masks")

for d in [SVG_DIR, PNG_DIR, MASK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------
# Simple NB→Mahadevan mapping placeholder
# --------------------------------------------
def nb_to_m(nb):
    return nb  # replace when you have true mapping


# --------------------------------------------
# Try to download historical SVG
# --------------------------------------------
def try_download_svg(m_num):
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/Indus_script_sign_{m_num:03d}.svg"
    resp = requests.get(url)

    if resp.status_code == 200 and resp.text.startswith("<svg"):
        out = SVG_DIR / f"sign_{m_num:03d}.svg"
        out.write_text(resp.text, encoding="utf-8")
        return out
    return None


# --------------------------------------------
# PURE PYTHON SVG → MASK (simple vector parser)
# --------------------------------------------
def svg_to_mask(svg_path, size=512):
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except:
        return None

    # Create blank image
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    # Parse <path d="...">
    for elem in root.iter():
        if "path" in elem.tag:
            d = elem.attrib.get("d", "")
            # Only handle simple M/L polygons
            pts = []
            tokens = d.replace(",", " ").split()

            i = 0
            while i < len(tokens):
                if tokens[i] in ("M", "L"):
                    x = float(tokens[i+1])
                    y = float(tokens[i+2])
                    pts.append((x, y))
                    i += 3
                else:
                    i += 1

            if len(pts) >= 2:
                draw.line(pts, fill=255, width=20)

    return mask


# --------------------------------------------
# Procedural glyph fallback
# --------------------------------------------
def procedural_glyph(nb_code, size=512):
    seed = int(hashlib.sha256(nb_code.encode()).hexdigest(), 16)
    img = Image.new("RGBA", (size, size), (255,255,255,0))
    draw = ImageDraw.Draw(img)
    cx, cy = size//2, size//2

    n = 4 + (seed % 4)
    for i in range(n):
        ang = (seed >> (i*3)) & 0xFF
        th = ang / 255.0 * 2 * math.pi
        r1 = size * (0.15 + i*0.1)
        x1 = cx + r1 * math.cos(th)
        y1 = cy + r1 * math.sin(th)
        x2 = cx + (r1*0.4) * math.cos(th + 0.7)
        y2 = cy + (r1*0.4) * math.sin(th + 0.7)
        draw.line([x1,y1,x2,y2], fill=(0,0,0,255), width=10)

    mask = img.convert("L").point(lambda p: 255 if p > 20 else 0)
    return img, mask


# --------------------------------------------
# Build registry
# --------------------------------------------
def build_registry(start=1, end=417):
    registry = {}

    for nb in range(start, end+1):
        nb_code = f"NB{nb:03d}"
        print(f"Processing {nb_code}")

        m_num = nb_to_m(nb)
        svg_path = try_download_svg(m_num)

        if svg_path:
            mask = svg_to_mask(svg_path)
            if mask:
                # save PNG for display
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

        # procedural fallback
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

    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    print("Registry built successfully.")


if __name__ == "__main__":
    build_registry()
