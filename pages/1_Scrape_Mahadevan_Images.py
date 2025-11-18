import streamlit as st
from PIL import Image
import requests
import io

st.title("View Mahadevan Page Images (No Saving Required)")

# Archive.org IIIF page image pattern
BASE_URL = (
    "https://archivelab.org/"
    "TheIndusScript.TextConcordanceAndTablesIravathanMahadevan/"
    "p{page}/full/full/0/default.jpg"
)

start_page = st.number_input("Start Page", 1, 600, 1)
end_page   = st.number_input("End Page", 1, 600, 10)

if st.button("Fetch & Display Pages"):
    st.info("Fetching page images from Archive.org…")
    progress = st.progress(0.0)

    total = end_page - start_page + 1

    for i, page in enumerate(range(start_page, end_page + 1)):
        url = BASE_URL.format(page=page)
        st.write(f"### Page {page}")
        st.write(url)

        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content))
                st.image(img, caption=f"Page {page}", use_column_width=True)
            else:
                st.warning(f"Page {page} not found:")
                st.code(url)

        except Exception as e:
            st.error(f"Failed to fetch page {page}")
            st.code(str(e))

        progress.progress((i+1)/total)

    st.success("Finished fetching pages.")
    st.info("You can now long-press or right-click each image to save it manually.")
