# streamlit_app.py
"""
Indus Holo — Admin / Metadata / Normalization page
Features:
 - Compact NB metadata editor (view / add / edit / delete)
 - Normalize uploaded or existing glyphs to consistent size & mask
 - Generate binary masks (threshold + morphological cleanup)
 - Safe JSON read/write with backups
 - Export normalized PNGs + registry to a simulator folder (optional)
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import json, io, shutil, datetime

st.set_page_config(layout="wide", page_title="Indus Holo — Admin")

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GLYPH_DIR = DATA_DIR / "glyphs" / "png"   # canonical storage for PNGs
MASK_DIR  = DATA_DIR / "glyphs" / "masks"
NB_JSON   = DATA_DIR / "nb_signs.json"    # canonical metadata registry
BACKUP_DIR= DATA_DIR / "backups"

# Ensure directories
for d in (DATA_DIR, GLYPH_DIR, MASK_DIR, BACKUP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Utilities ----------
def safe_load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to parse JSON {path}: {e}")
        return {}

def safe_write_json(path, data):
    # write with backup
    if path.exists():
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        bak = BACKUP_DIR / f"{path.name}.{ts}.bak"
        shutil.copy2(path, bak)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def list_nb_files():
    files = sorted(GLYPH_DIR.glob("NB*.png"))
    return [p.name for p in files]

def normalize_image(pil_img, output_size=256, pad=8, bg=255):
    """
    Normalize glyph image:
     - convert to grayscale
     - autocontrast
     - threshold to mask
     - crop to content bounding box
     - scale to fit output_size - 2*pad
     - center on output_size canvas (bg white)
    Returns normalized PIL image (L).
    """
    im = pil_img.convert("L")
    im = ImageOps.autocontrast(im)
    # create binary mask
    bw = im.point(lambda p: 0 if p < 200 else 255).convert("L")  # 200 is conservative, adjustable
    arr = np.array(bw)
    ys, xs = np.nonzero(arr == 0)  # ink = black (0)
    if len(xs) == 0 or len(ys) == 0:
        # nothing detected — just resize
        out = im.resize((output_size, output_size), Image.LANCZOS)
        return out
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    # add a small margin
    margin = max(2, int(min((maxx-minx,maxy-miny)*0.05, 10)))
    minx = max(0, minx - margin)
    miny = max(0, miny - margin)
    maxx = min(im.width-1, maxx + margin)
    maxy = min(im.height-1, maxy + margin)
    crop = im.crop((minx, miny, maxx+1, maxy+1))
    # scale to fit
    target = output_size - 2*pad
    w,h = crop.size
    scale = min(target / w, target / h)
    new_w = max(1, int(w*scale))
    new_h = max(1, int(h*scale))
    resized = crop.resize((new_w, new_h), Image.LANCZOS)
    # center on canvas
    canvas = Image.new("L", (output_size, output_size), color=bg)
    xoff = (output_size - new_w)//2
    yoff = (output_size - new_h)//2
    canvas.paste(resized, (xoff, yoff))
    return canvas

def generate_mask_from_image(pil_img, threshold=128, blur_radius=1):
    """
    Generate a binary mask (0 background, 255 ink) from an image.
    - blur optional to remove speckle
    - threshold to create mask
    - small morphological cleanup via PIL filters
    """
    im = pil_img.convert("L")
    if blur_radius > 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.array(im)
    mask_arr = (arr < threshold).astype(np.uint8) * 255  # ink=255
    mask = Image.fromarray(mask_arr).convert("L")
    # small morphological cleanup: remove small islands via median
    mask = mask.filter(ImageFilter.MedianFilter(size=3))
    return mask

# ---------- UI ----------
st.title("Indus Holo — Registry & Normalization Admin")
st.markdown("Use this page to edit NB metadata, normalize glyph PNGs, create masks, and export to your simulator folder.")

# Tabs: registry editor, batch normalize, single normalize/upload, export
tab = st.tabs(["Registry Editor", "Batch Normalize & Mask", "Upload & Normalize (one)", "Export to Simulator"])

# -------------------- Registry Editor --------------------
with tab[0]:
    st.header("Registry Editor — view / edit NB metadata")
    nb_data = safe_load_json(NB_JSON)

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Existing NB PNGs")
        nb_list = list_nb_files()
        selected = st.selectbox("Select an NB PNG file", ["--none--"] + nb_list)
        if selected and selected != "--none--":
            nb_code = Path(selected).stem  # e.g. NB001
        else:
            nb_code = None

        st.markdown("Or enter NB code to edit/add")
        typed_nb = st.text_input("NB code (e.g. NB001)", value=(nb_code or "NB001"))
        if typed_nb:
            typed_nb = typed_nb.strip().upper()

    with col2:
        st.subheader("Metadata editor")
        if typed_nb:
            meta = nb_data.get(typed_nb, {
                "nb": typed_nb,
                "label": "",
                "mahadevan": None,
                "variants": [],
                "notes": "",
                "site": "",
                "frequency_class": None,
                "source": None
            })
            # show image preview if exists
            candidate_img = None
            p_candidate = GLYPH_DIR / f"{typed_nb}.png"
            if p_candidate.exists():
                try:
                    candidate_img = Image.open(p_candidate)
                except:
                    candidate_img = None
            if candidate_img:
                st.image(candidate_img, width=160, caption=f"{typed_nb}.png")
            # editable fields
            meta["label"] = st.text_input("Label (short)", value=meta.get("label",""))
            meta["mahadevan"] = st.text_input("Mahadevan index (if known)", value=str(meta.get("mahadevan","") or ""))
            meta["site"] = st.text_input("Site / findspot", value=meta.get("site","") or "")
            meta["frequency_class"] = st.selectbox("Frequency class (research tag)", ["Unknown","Economic","Social","Cosmic","Deity","Other"], index=0 if not meta.get("frequency_class") else ["Unknown","Economic","Social","Cosmic","Deity","Other"].index(meta.get("frequency_class","Unknown")))
            meta["notes"] = st.text_area("Notes", value=meta.get("notes",""))
            meta["source"] = st.text_input("Source (file/URL)", value=meta.get("source","") or "")

            col_actions = st.columns(3)
            if col_actions[0].button("Save metadata (update JSON)"):
                nb_data[typed_nb] = meta
                safe_write_json(NB_JSON, nb_data)
                st.success(f"Saved metadata for {typed_nb} to {NB_JSON}")

            if col_actions[1].button("Delete metadata entry"):
                if typed_nb in nb_data:
                    nb_data.pop(typed_nb, None)
                    safe_write_json(NB_JSON, nb_data)
                    st.success(f"Deleted metadata for {typed_nb}")

            if col_actions[2].button("Create placeholder PNG (procedural)"):
                # generate a simple procedural placeholder normalized image and mask
                proc = Image.new("L", (256,256), 255)
                # simple cross
                draw = ImageOps.autocontrast(proc)
                proc = normalize_image(proc, output_size=256)
                outp = GLYPH_DIR / f"{typed_nb}.png"
                proc.save(outp)
                st.success(f"Created placeholder {outp}")
        else:
            st.info("Enter an NB code to edit or select an existing file from left.")

# -------------------- Batch Normalize & Mask --------------------
with tab[1]:
    st.header("Batch Normalize & Mask Generation")
    st.write("This will normalize all PNGs in the glyph folder and (re)generate masks. It does not overwrite your original images without confirmation.")
    st.markdown("**Options**")

    batch_size = st.number_input("Process first N files (0 = all)", min_value=0, value=0)
    out_size = st.number_input("Output size (px)", min_value=32, value=256)
    threshold = st.slider("Mask threshold (0-255; lower = more ink)", 0, 255, 180)
    blur = st.slider("Mask blur radius", 0, 4, 1)
    dryrun = st.checkbox("Dry run (show what would be done)", value=True)

    files = list_nb_files()
    if batch_size > 0:
        files = files[:batch_size]

    st.write(f"Found {len(files)} PNGs to consider in {GLYPH_DIR}")

    if st.button("Run Batch Normalize & Mask"):
        processed = 0
        for fn in files:
            p = GLYPH_DIR / fn
            try:
                pil = Image.open(p)
                norm = normalize_image(pil, output_size=out_size)
                mask = generate_mask_from_image(norm, threshold=threshold, blur_radius=blur)
                # preview first few if dryrun
                if dryrun and processed < 6:
                    st.image(norm, caption=f"Normalized preview {fn}", width=120)
                    st.image(mask, caption=f"Mask preview {fn}", width=120)
                if not dryrun:
                    # overwrite normalized image and write mask
                    norm.save(p)
                    mask.save(MASK_DIR / fn)
                processed += 1
            except Exception as e:
                st.warning(f"Failed processing {fn}: {e}")
        st.success(f"Batch done. Processed {processed} files. Dryrun={dryrun}")

# -------------------- Single Upload & Normalize --------------------
with tab[2]:
    st.header("Upload a glyph PNG/JPG and normalize (one-by-one)")
    uploaded = st.file_uploader("Upload glyph image (PNG/JPG)", type=["png","jpg","jpeg"])
    target_nb = st.text_input("Assign NB code to this upload (e.g. NB123)", value="NB000")
    size_out = st.number_input("Output size (px)", min_value=32, value=256)
    mask_threshold = st.slider("Mask threshold", 0, 255, 180)
    mask_blur = st.slider("Mask blur radius", 0, 4, 1)

    if uploaded:
        pil = Image.open(uploaded)
        st.image(pil, caption="Uploaded (raw)", width=240)
        norm = normalize_image(pil, output_size=size_out)
        st.image(norm, caption="Normalized preview", width=240)
        mask = generate_mask_from_image(norm, threshold=mask_threshold, blur_radius=mask_blur)
        st.image(mask, caption="Generated mask preview", width=240)

        col_ok = st.columns(2)
        if col_ok[0].button("Save normalized PNG & mask"):
            nbid = target_nb.strip().upper() or "NB000"
            out_png = GLYPH_DIR / f"{nbid}.png"
            out_mask = MASK_DIR / f"{nbid}.png"
            norm.save(out_png)
            mask.save(out_mask)
            st.success(f"Saved {out_png} and {out_mask}")
        if col_ok[1].button("Save only to temp and download"):
            buf = io.BytesIO()
            norm.save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download normalized PNG", data=buf.read(), file_name=f"{target_nb}.png", mime="image/png")

# -------------------- Export to Simulator --------------------
with tab[3]:
    st.header("Export normalized glyphs & registry to simulator folder")
    st.write("If you have a simulator package or another repo layout, you can copy normalized PNGs + masks + registry JSON to a destination folder.")
    dest = st.text_input("Destination folder path (absolute or relative)", value=str(ROOT / "simulator" / "data"))
    btn = st.button("Export now (copies PNGs, masks, nb_signs.json)")
    if btn:
        destp = Path(dest)
        try:
            destp.mkdir(parents=True, exist_ok=True)
            # copy PNGs
            copied = 0
            for p in GLYPH_DIR.glob("NB*.png"):
                shutil.copy2(p, destp / p.name)
                copied += 1
            # copy masks
            m_copied = 0
            for m in MASK_DIR.glob("NB*.png"):
                shutil.copy2(m, destp / m.name)
                m_copied += 1
            # copy registry
            if NB_JSON.exists():
                shutil.copy2(NB_JSON, destp / NB_JSON.name)
            st.success(f"Export complete. Copied {copied} PNGs and {m_copied} masks to {destp}")
        except Exception as e:
            st.error(f"Export failed: {e}")

st.markdown("---")
st.caption("This page helps you curate and normalize your historical glyph collection. After exporting to the simulator path, the self-contained simulator app can load the PNGs and updated nb_signs.json automatically.")
