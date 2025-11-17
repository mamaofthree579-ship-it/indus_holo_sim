# pages/1_Extract_Mahadevan.py
import streamlit as st
from pathlib import Path
import requests
import traceback
import os
import zipfile

st.title("Mahadevan PDF Extractor — Temp-Safe Version")

# ---------------------------------------------------------
# Use /tmp/ for everything (always writable in Streamlit)
# ---------------------------------------------------------
TMP = Path("/tmp")
PDF_PATH = TMP / "mahadevan_full.pdf"
OUT_ROOT = TMP / "mahadevan_extracted"
PAGES_DIR = OUT_ROOT / "pages_png"
EMBED_DIR = OUT_ROOT / "embedded_images"

for d in [OUT_ROOT, PAGES_DIR, EMBED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ARCHIVE_URL = (
    "https://archive.org/download/"
    "TheIndusScript.TextConcordanceAndTablesIravathanMahadevan/"
    "The%20Indus%20Script.%20Text%2C%20Concordance%20and%20Tables%20"
    "-Iravathan%20Mahadevan.pdf"
)

st.write("Download location:", PDF_PATH)

# ---------------------------------------------------------
# Download PDF if not present
# ---------------------------------------------------------
if not PDF_PATH.exists():
    st.info("Downloading Mahadevan PDF to /tmp/ …")
    try:
        with requests.get(ARCHIVE_URL, stream=True, timeout=90) as r:
            r.raise_for_status()
            with open(PDF_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success("PDF downloaded to /tmp/")
    except Exception:
        st.error("Download failed:")
        st.text(traceback.format_exc())
        st.stop()
else:
    st.success("PDF already downloaded.")

# ---------------------------------------------------------
# Extraction
# ---------------------------------------------------------
if st.button("Extract Pages and Embedded Glyphs"):
    st.info("Extracting… This may take a few minutes.")
    progress = st.progress(0)

    try:
        try:
            import fitz  # PyMuPDF
            st.write("Using PyMuPDF…")

            doc = fitz.open(str(PDF_PATH))
            total_pages = len(doc)

            for i in range(total_pages):
                page = doc[i]

                # extract embedded images
                image_list = page.get_images(full=True)
                if image_list:
                    for j, img in enumerate(image_list):
                        xref = img[0]
                        base_img = doc.extract_image(xref)
                        bytes_ = base_img["image"]
                        ext = base_img.get("ext", "png")

                        out_img = EMBED_DIR / f"page{i+1:03d}_img{j+1}.{ext}"
                        with open(out_img, "wb") as f:
                            f.write(bytes_)

                # render page
                pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                out_page = PAGES_DIR / f"page_{i+1:03d}.png"
                pix.save(str(out_page))

                progress.progress((i+1)/total_pages)

            st.success("Extraction complete!")
        except Exception as fitz_err:
            st.warning("PyMuPDF failed. Trying pdf2image…")
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(str(PDF_PATH), dpi=150)
                total = len(pages)
                for i, im in enumerate(pages):
                    out_page = PAGES_DIR / f"page_{i+1:03d}.png"
                    im.save(out_page)
                    progress.progress((i+1)/total)
                st.success("pdf2image extraction complete!")
            except Exception as pdf_err:
                st.error("Both extractors failed.")
                st.text("PYMUPDF ERROR:\n" + str(fitz_err))
                st.text("PDF2IMAGE ERROR:\n" + str(pdf_err))
                st.stop()

    except Exception as e:
        st.error("Unexpected extraction error:")
        st.text(traceback.format_exc())
        st.stop()


# ---------------------------------------------------------
# Previews
# ---------------------------------------------------------
st.write("---")
st.subheader("Preview Extracted Pages")

page_files = sorted(PAGES_DIR.glob("*.png"))
if page_files:
    cols = st.columns(3)
    for i, p in enumerate(page_files[:9]):
        with cols[i % 3]:
            st.image(str(p), caption=p.name, use_column_width=True)

st.subheader("Preview Embedded Images")

embed_files = sorted(EMBED_DIR.glob("*.*"))
if embed_files:
    cols = st.columns(3)
    for i, p in enumerate(embed_files[:9]):
        with cols[i % 3]:
            st.image(str(p), caption=p.name, use_column_width=True)

# ---------------------------------------------------------
# ZIP Download
# ---------------------------------------------------------
if st.button("Download All Extracted Pages (ZIP)"):
    zip_path = TMP / "mahadevan_pages.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for p in PAGES_DIR.glob("*.png"):
            z.write(p, p.name)

    with open(zip_path, "rb") as f:
        st.download_button(
            "Download ZIP",
            f.read(),
            file_name="mahadevan_pages.zip",
            mime="application/zip"
        )
