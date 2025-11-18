# pages/0_Simulator_Admin.py
"""
Indus Holo — Admin / Metadata / Normalization page
(Updated with unique Streamlit keys to avoid DuplicateElementId errors)
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
GLYPH_DIR = DATA_DIR / "glyphs" / "png"
MASK_DIR  = DATA_DIR / "glyphs" / "masks"
NB_JSON   = DATA_DIR / "nb_signs.json"
BACKUP_DIR= DATA_DIR / "backups"

for d in (DATA_DIR, GLYPH_DIR, MASK_DIR, BACKUP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Utility Functions ----------
def safe_load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to parse JSON {path}: {e}")
        return {}

def safe_write_json(path, data):
    if path.exists():
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        bak = BACKUP_DIR / f"{path.name}.{ts}.bak"
        shutil.copy2(path, bak)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def list_nb_files():
    return sorted(GLYPH_DIR.glob("NB*.png"))

def normalize_image(pil_img, output_size=256, pad=8, bg=255):
    im = pil_img.convert("L")
    im = ImageOps.autocontrast(im)
    bw = im.point(lambda p: 0 if p < 200 else 255)
    arr = np.array(bw)
    ys, xs = np.nonzero(arr == 0)
    if len(xs) == 0:
        return im.resize((output_size, output_size), Image.LANCZOS)
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    margin = max(2, int(min((maxx-minx, maxy-miny)) * 0.05))
    crop = im.crop((max(minx-margin,0), max(miny-margin,0), min(maxx+margin,im.width), min(maxy+margin,im.height)))
    w,h = crop.size
    target = output_size - 2*pad
    scale = min(target/w, target/h)
    resized = crop.resize((max(1,int(w*scale)), max(1,int(h*scale))), Image.LANCZOS)
    canvas = Image.new("L", (output_size, output_size), bg)
    canvas.paste(resized, ((output_size-resized.width)//2, (output_size-resized.height)//2))
    return canvas

def generate_mask_from_image(pil_img, threshold=128, blur_radius=1):
    im = pil_img.convert("L")
    if blur_radius > 0:
        im = im.filter(ImageFilter.GaussianBlur(blur_radius))
    arr = np.array(im)
    mask = (arr < threshold).astype(np.uint8) * 255
    mask = Image.fromarray(mask).filter(ImageFilter.MedianFilter(3))
    return mask

# ---------- Layout ----------
st.title("Indus Holo — Registry & Normalization Admin")


# ----- NAVIGATION -----
st.sidebar.markdown("## 🔀 Navigation")
st.sidebar.page_link("streamlit_app.py", label="🔮 Main Simulator")
st.sidebar.page_link("pages/0_Simulator_Admin.py", label="⚙️ Admin / Normalization")
st.sidebar.page_link("pages/7_Diffraction_Lab.py", label="🌈 Diffraction Physics Lab")
# -----------------------

tab = st.tabs(["Registry Editor", "Batch Normalize & Mask", "Upload & Normalize (one)", "Export to Simulator"])

# ==================================================================
# TAB 1 — REGISTRY EDITOR
# ==================================================================
with tab[0]:
    st.header("Registry Editor — view / edit NB metadata")

    nb_data = safe_load_json(NB_JSON)

    col1, col2 = st.columns([1,2])

    with col1:
        st.subheader("Available NB glyphs")
        nb_files = list_nb_files()
        nb_select = st.selectbox("Select glyph", ["--none--"] + [p.name for p in nb_files], key="reg_nb_pick")

        inferred_nb = nb_select.replace(".png","") if nb_select != "--none--" else ""
        typed_nb = st.text_input("NB code", inferred_nb, key="reg_nb_code").upper().strip()

    with col2:
        if typed_nb:
            st.subheader(f"Metadata for {typed_nb}")

            default_meta = {
                "nb": typed_nb,
                "label": "",
                "mahadevan": "",
                "variants": [],
                "site": "",
                "frequency_class": "Unknown",
                "notes": "",
                "source": ""
            }
            meta = nb_data.get(typed_nb, default_meta)

            # Editable fields
            meta["label"] = st.text_input("Label", meta.get("label",""), key="reg_label")
            meta["mahadevan"] = st.text_input("Mahadevan index", meta.get("mahadevan",""), key="reg_mahadevan")
            meta["site"] = st.text_input("Site", meta.get("site",""), key="reg_site")
            meta["frequency_class"] = st.selectbox(
                "Frequency Class",
                ["Unknown","Economic","Social","Cosmic","Deity","Other"],
                index=["Unknown","Economic","Social","Cosmic","Deity","Other"].index(meta.get("frequency_class","Unknown")),
                key="reg_freq"
            )
            meta["notes"] = st.text_area("Notes", meta.get("notes",""), key="reg_notes")
            meta["source"] = st.text_input("Source", meta.get("source",""), key="reg_source")

            img_path = GLYPH_DIR / f"{typed_nb}.png"
            if img_path.exists():
                st.image(Image.open(img_path), width=150)

            c1, c2, c3 = st.columns(3)
            if c1.button("Save", key="reg_save"):
                nb_data[typed_nb] = meta
                safe_write_json(NB_JSON, nb_data)
                st.success("Saved.")

            if c2.button("Delete", key="reg_delete"):
                nb_data.pop(typed_nb, None)
                safe_write_json(NB_JSON, nb_data)
                st.success("Deleted from registry.")

            if c3.button("Create placeholder PNG", key="reg_placeholder"):
                img = Image.new("L",(256,256),255)
                img.save(GLYPH_DIR / f"{typed_nb}.png")
                st.success("Placeholder created.")

# ==================================================================
# TAB 2 — BATCH NORMALIZE
# ==================================================================
with tab[1]:
    st.header("Batch Normalize & Mask Generation")

    batch_limit = st.number_input("Process N files (0 = all)", min_value=0, value=0, key="batch_limit")
    batch_size = st.number_input("Output size", min_value=32, value=256, key="batch_size")

    thresh = st.slider("Mask threshold", 0, 255, 180, key="batch_thresh")
    blur_r = st.slider("Mask blur", 0, 5, 1, key="batch_blur")
    dryrun = st.checkbox("Dry run (preview only)", value=True, key="batch_dryrun")

    files = list_nb_files()
    if batch_limit > 0:
        files = files[:batch_limit]

    if st.button("Run batch", key="batch_run"):
        count = 0
        for p in files:
            try:
                src = Image.open(p)
                norm = normalize_image(src, output_size=batch_size)
                mask = generate_mask_from_image(norm, threshold=thresh, blur_radius=blur_r)

                if dryrun and count < 6:
                    st.image(norm, width=120, caption=f"{p.name} normalized")
                    st.image(mask, width=120, caption="mask")

                if not dryrun:
                    norm.save(p)
                    mask.save(MASK_DIR / p.name)

                count += 1
            except Exception as e:
                st.warning(f"Failed on {p.name}: {e}")

        st.success(f"Processed {count} files. Dry run={dryrun}")

# ==================================================================
# TAB 3 — SINGLE UPLOAD
# ==================================================================
with tab[2]:
    st.header("Upload + Normalize (single glyph)")

    uploaded = st.file_uploader("Upload PNG/JPG", type=["png","jpg","jpeg"], key="single_upload")
    nb_assign = st.text_input("Assign NB###", value="NB000", key="single_nb").upper().strip()
    out_size = st.number_input("Output size", min_value=32, value=256, key="single_size")
    th = st.slider("Mask threshold", 0, 255, 180, key="single_thresh")
    bl = st.slider("Mask blur", 0, 5, 1, key="single_blur")

    if uploaded:
        pil = Image.open(uploaded)
        st.image(pil, width=200, caption="Uploaded")

        norm = normalize_image(pil, output_size=out_size)
        st.image(norm, width=200, caption="Normalized")

        mask = generate_mask_from_image(norm, threshold=th, blur_radius=bl)
        st.image(mask, width=200, caption="Mask")

        c1, c2 = st.columns(2)
        if c1.button("Save to data/", key="single_save"):
            norm.save(GLYPH_DIR / f"{nb_assign}.png")
            mask.save(MASK_DIR / f"{nb_assign}.png")
            st.success("Saved.")

        if c2.button("Download normalized PNG", key="single_download"):
            buf = io.BytesIO()
            norm.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), file_name=f"{nb_assign}.png")

# ==================================================================
# TAB 4 — EXPORT
# ==================================================================
with tab[3]:
    st.header("Export normalized PNGs + masks + registry → simulator folder")

    dest = st.text_input("Destination folder", value=str(ROOT/"simulator"/"data"), key="export_dest")

    if st.button("Export now", key="export_btn"):
        destp = Path(dest)
        destp.mkdir(parents=True, exist_ok=True)

        n_png = 0
        n_mask = 0
        for p in GLYPH_DIR.glob("NB*.png"):
            shutil.copy2(p, destp / p.name)
            n_png += 1
        for m in MASK_DIR.glob("NB*.png"):
            shutil.copy2(m, destp / m.name)
            n_mask += 1
        if NB_JSON.exists():
            shutil.copy2(NB_JSON, destp / NB_JSON.name)

        st.success(f"Export complete: {n_png} PNGs + {n_mask} masks + registry copied.")

st.caption("Unique keys added to prevent Streamlit DuplicateElementId errors.")
