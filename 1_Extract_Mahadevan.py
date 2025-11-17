# pages/1_Extract_Mahadevan.py
import streamlit as st
from pathlib import Path
import traceback
import io

st.title("Mahadevan PDF — Page & Embedded Image Extractor")

# Path to the Mahadevan PDF as uploaded in the repo
PDF_PATH = Path("data") / "The Indus Script. Text, Concordance and Tables -Iravatham Mahadevan.pdf"
OUT_ROOT = Path("data") / "mahadevan_extracted"
PAGES_DIR = OUT_ROOT / "pages_png"
EMBED_DIR = OUT_ROOT / "embedded_images"

for d in (OUT_ROOT, PAGES_DIR, EMBED_DIR):
    d.mkdir(parents=True, exist_ok=True)

st.write("PDF path:", PDF_PATH)

if not PDF_PATH.exists():
    st.error("Mahadevan PDF not found at data/… Make sure the file is present.")
    st.stop()

if st.button("Extract pages & embedded images"):
    st.info("Starting extraction. This may take a minute for the whole pdf.")
    progress = st.progress(0)
    try:
        # Try PyMuPDF first
        try:
            import fitz
            doc = fitz.open(str(PDF_PATH))
            page_count = len(doc)
            for i in range(page_count):
                page = doc[i]
                # extract embedded images
                img_list = page.get_images(full=True)
                if img_list:
                    for ii, img in enumerate(img_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        ext = base_image.get("ext", "png")
                        out_path = EMBED_DIR / f"page{ i+1 :03d }_img{ ii+1 }.{ext}"
                        with open(out_path, "wb") as f:
                            f.write(image_bytes)
                # render page
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                page_png = PAGES_DIR / f"page_{i+1:03d}.png"
                pix.save(str(page_png))
                progress.progress((i+1)/page_count)
            st.success(f"Rendered {page_count} pages and saved embedded images (if any).")
        except Exception as e_fitz:
            st.warning("PyMuPDF not available or failed; trying pdf2image fallback.")
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(str(PDF_PATH), dpi=150)
                for i, pil_page in enumerate(pages):
                    pth = PAGES_DIR / f"page_{i+1:03d}.png"
                    pil_page.save(pth)
                    progress.progress((i+1)/len(pages))
                st.success(f"Rendered {len(pages)} pages using pdf2image.")
            except Exception as e_pdf2:
                st.error("Both PyMuPDF and pdf2image failed. See exceptions below.")
                st.text(traceback.format_exc())
    except Exception as e:
        st.error("Extraction failed. See exception.")
        st.text(traceback.format_exc())

# show samples and download links
st.write("Pages directory:", PAGES_DIR)
page_files = sorted(PAGES_DIR.glob("*.png"))
embed_files = sorted(EMBED_DIR.glob("*.*"))
if page_files:
    st.subheader("Sample page images")
    cols = st.columns(3)
    for idx, p in enumerate(page_files[:9]):
        with cols[idx % 3]:
            st.image(str(p), caption=p.name, use_column_width=True)
if embed_files:
    st.subheader("Embedded images (if any)")
    for e in embed_files[:12]:
        st.image(str(e), caption=e.name, use_column_width=True)

st.write("---")
st.write("Extraction outputs are saved under data/mahadevan_extracted/ . You can now run the glyph-crop step.")
