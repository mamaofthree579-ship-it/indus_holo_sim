import streamlit as st
from pathlib import Path
from PIL import Image
import json

st.title("Map Cropped Glyphs → NB Numbers")

TMP = Path("/tmp")
CROPS = TMP / "glyph_candidates"
NB_DIR = TMP / "nb_glyphs"
NB_DIR.mkdir(parents=True, exist_ok=True)

SIGNS_JSON = TMP / "nb_signs.json"

# ensure signs json exists
if not SIGNS_JSON.exists():
    SIGNS_JSON.write_text(json.dumps({}, indent=2))

signs = json.loads(SIGNS_JSON.read_text())

crop_files = sorted(CROPS.glob("*.png"))
if not crop_files:
    st.error("No glyph candidates found. Run Auto-Crop first.")
    st.stop()

choice = st.selectbox("Choose a glyph crop", [p.name for p in crop_files])
cp = CROPS / choice

st.image(str(cp), caption="Selected glyph", use_column_width=True)

nb_number = st.number_input("NB Number", min_value=1, max_value=1000, value=1)
nb_id = f"NB{nb_number:03d}"

if st.button("Assign this glyph to NB"):
    img = Image.open(cp)

    save_path = NB_DIR / f"{nb_id}.png"
    img.save(save_path)

    # update JSON
    signs[nb_id] = {
        "glyph_path": str(save_path),
        "source": "historical",
        "notes": "Mapped from auto-crop"
    }

    SIGNS_JSON.write_text(json.dumps(signs, indent=2))

    st.success(f"Assigned {choice} → {nb_id}")
    st.write("Saved:", save_path)
    st.json(signs.get(nb_id))
