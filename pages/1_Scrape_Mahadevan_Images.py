import streamlit as st
from pathlib import Path
import requests
from PIL import Image
import io, zipfile

st.title("Scrape Mahadevan Page Images from Archive.org")

# Where images will be saved
OUT_DIR = Path("/tmp/mahadevan_pages")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Base URL pattern
BASE_URL = (
    "https://iiif.archivelab.org/iiif/"
    "TheIndusScript.TextConcordanceAndTablesIravathanMahadevan/"
    "p{page}/full/full/0/default.jpg"
)

start_page = st.number_input("Start Page", 1, 600, 1)
end_page   = st.number_input("End Page", 1, 600, 20)

if st.button("Scrape Pages"):
    st.info("Downloading page images…")
    saved = []
    total = end_page - start_page + 1
    progress = st.progress(0.0)

    for i, p in enumerate(range(start_page, end_page+1)):
        url = BASE_URL.format(page=p)
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content))
                out = OUT_DIR / f"page_{p:03d}.jpg"
                img.save(out)
                saved.append(out)
        except:
            pass

        progress.progress((i+1)/total)

    st.success(f"Downloaded {len(saved)} page images into /tmp/mahadevan_pages/")
    if saved:
        st.image(str(saved[0]), caption="Example page")

        # Let user download all pages
        zip_path = Path("/tmp/mahadevan_pages.zip")
        with zipfile.ZipFile(zip_path, "w") as z:
            for s in saved:
                z.write(s, s.name)

        with open(zip_path, "rb") as f:
            st.download_button(
                "Download ZIP",
                f.read(),
                "mahadevan_pages.zip",
                mime="application/zip"
            )
