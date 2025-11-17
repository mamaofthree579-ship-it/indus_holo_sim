import streamlit as st
from pathlib import Path
import requests, traceback, io, zipfile
from pypdf import PdfReader
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

st.title("Mahadevan PDF Extractor — Pure Python Mode")

TMP = Path("/tmp")
PDF_PATH = TMP / "mahadevan_full.pdf"

OUT_ROOT = TMP / "mahadevan_extracted"
SVG_DIR = OUT_ROOT / "pages_svg"
PNG_DIR = OUT_ROOT / "pages_png"

for d in (OUT_ROOT, SVG_DIR, PNG_DIR):
    d.mkdir(parents=True, exist_ok=True)

ARCHIVE_URL = (
    "https://archive.org/download/"
    "TheIndusScript.TextConcordanceAndTablesIravathanMahadevan/"
    "The%20Indus%20Script.%20Text%2C%20Concordance%20and%20Tables%20-Iravathan%20Mahadevan.pdf"
)

# ----------------------------------------------------------
# Download PDF
# ----------------------------------------------------------
if not PDF_PATH.exists():
    st.info("Downloading Mahadevan PDF…")
    try:
        r = requests.get(ARCHIVE_URL, stream=True)
        with open(PDF_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        st.success("PDF downloaded.")
    except Exception as e:
        st.error("Download failed.")
        st.text(traceback.format_exc())
        st.stop()
else:
    st.success("PDF already downloaded.")

# ----------------------------------------------------------
# Convert PDF pages → SVG → PNG
# ----------------------------------------------------------
def convert_page_to_svg(page, output_svg):
    """
    Very simplified SVG wrapper: writes text-only content.
    We will still get page layout, images may not convert.
    """
    content = page.extract_text() or ""
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1600">
        <style> text {{ font-size:18px; font-family:Helvetica; }} </style>
        <text x="50" y="50">{content.replace("&","&amp;").replace("<","&lt;")}</text>
    </svg>
    """
    output_svg.write_text(svg)

if st.button("Extract Pages"):
    try:
        reader = PdfReader(str(PDF_PATH))
        total = len(reader.pages)
        progress = st.progress(0.0)

        for i, page in enumerate(reader.pages):
            svg_path = SVG_DIR / f"page_{i+1:03d}.svg"
            png_path = PNG_DIR / f"page_{i+1:03d}.png"

            # Create SVG version
            convert_page_to_svg(page, svg_path)

            # Render to PNG
            drawing = svg2rlg(str(svg_path))
            renderPM.drawToFile(
                drawing,
                str(png_path),
                fmt="PNG",
                dpi=150
            )

            progress.progress((i+1) / total)

        st.success(f"Extracted {total} pages into /tmp/")

    except Exception as e:
        st.error("Extraction failed:")
        st.text(traceback.format_exc())

# ----------------------------------------------------------
# Previews
# ----------------------------------------------------------
png_files = list(PNG_DIR.glob("*.png"))
if png_files:
    st.subheader("Preview Page Images")
    for p in png_files[:6]:
        st.image(str(p), caption=p.name)

# ----------------------------------------------------------
# Download ZIP
# ----------------------------------------------------------
if png_files:
    if st.button("Download all as ZIP"):
        zip_path = TMP / "mahadevan_pages.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            for p in png_files:
                z.write(p, p.name)

        with open(zip_path, "rb") as f:
            st.download_button(
                "Download ZIP",
                f.read(),
                file_name="mahadevan_pages.zip"
            )
