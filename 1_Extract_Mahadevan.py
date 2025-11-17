# pages/1_Extract_Mahadevan.py
import streamlit as st
from pathlib import Path
import requests
import traceback

st.title("Mahadevan PDF Extractor — Page Images & Embedded Glyphs")

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
PDF_PATH = DATA / "mahadevan_full.pdf"

OUT_ROOT = DATA / "mahadevan_extracted"
PAGES_DIR = OUT_ROOT / "pages_png"
EMBED_DIR = OUT_ROOT / "embedded_images"

for d in [DATA, OUT_ROOT, PAGES_DIR, EMBED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# PDF Download from Archive.org
# ---------------------------------------------------------
ARCHIVE_URL = (
    "https://archive.org/download/"
    "TheIndusScript.TextConcordanceAndTablesIravathanMahadevan/"
    "The%20Indus%20Script.%20Text%2C%20Concordance%20and%20Tables%20"
    "-Iravathan%20Mahadevan.pdf"
)

if not PDF_PATH.exists():
    st.info("Mahadevan PDF not found. Downloading from Archive.org…")
    try:
        with requests.get(ARCHIVE_URL, stream=True, timeout=90) as r:
            r.raise_for_status()
            with open(PDF_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success("PDF downloaded successfully!")
    except Exception:
        st.error("PDF download failed. See details below.")
        st.text(traceback.format_exc())
        st.stop()
else:
    st.success("Mahadevan PDF already downloaded.")

st.write("PDF Path:", PDF_PATH)

# ---------------------------------------------------------
# Begin extraction: tries PyMuPDF, then pdf2image
# ---------------------------------------------------------
if st.button("Extract Pages and Embedded Glyph Images"):
    st.info("Starting extraction… this may take a few minutes.")
    progress = st.progress(0)

    try:
        try:
            import fitz  # PyMuPDF
            st.write("Using PyMuPDF renderer…")

            doc = fitz.open(str(PDF_PATH))
            total_pages = len(doc)

            for i in range(total_pages):
                page = doc[i]

                # Extract embedded images
                img_list = page.get_images(full=True)
                if img_list:
                    for j, img in enumerate(img_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        ext = base_image.get("ext", "png")

                        out_img = EMBED_DIR / f"page{i+1:03d}_img{j+1}.{ext}"
                        with open(out_img, "wb") as f:
                            f.write(img_bytes)

                # Render pages
                zoom = 2.0  # 144 DPI approx
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                out_page = PAGES_DIR / f"page_{i+1:03d}.png"
                pix.save(str(out_page))

                progress.progress((i + 1) / total_pages)

            st.success(f"Extracted {total_pages} pages with PyMuPDF!")

        except Exception as pymupdf_err:
            st.warning("PyMuPDF failed. Trying pdf2image fallback…")

            try:
                from pdf2image import convert_from_path

                pages = convert_from_path(str(PDF_PATH), dpi=150)
                total_pages = len(pages)

                for i, page_img in enumerate(pages):
                    out_page = PAGES_DIR / f"page_{i+1:03d}.png"
                    page_img.save(out_page)
                    progress.progress((i + 1) / total_pages)

                st.success(f"Extracted {total_pages} pages using pdf2image!")

            except Exception as pdf2_err:
                st.error("Both extractors failed.")
                st.text("PyMuPDF error:\n" + str(pymupdf_err))
                st.text("pdf2image error:\n" + str(pdf2_err))
                st.stop()

    except Exception:
        st.error("Extraction error occurred:")
        st.text(traceback.format_exc())
        st.stop()


# ---------------------------------------------------------
# Sample previews
# ---------------------------------------------------------
st.write("---")
st.subheader("Preview Extracted Page Images")

page_files = sorted(PAGES_DIR.glob("*.png"))
if page_files:
    cols = st.columns(3)
    for i, p in enumerate(page_files[:9]):
        with cols[i % 3]:
            st.image(str(p), caption=p.name, use_column_width=True)

st.subheader("Preview Embedded Glyph Images (if any)")

embed_files = sorted(EMBED_DIR.glob("*.*"))
if embed_files:
    cols = st.columns(3)
    for i, p in enumerate(embed_files[:9]):
        with cols[i % 3]:
            st.image(str(p), caption=p.name, use_column_width=True)

st.write("Extraction folders:")
st.write("• Page PNGs:", PAGES_DIR)
st.write("• Embedded glyphs:", EMBED_DIR)
