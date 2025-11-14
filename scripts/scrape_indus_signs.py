#!/usr/bin/env python3
"""
scrape_indus_signs.py

Robust scraper to assemble a nb_signs.json registry (NB001..NB417) and optionally
download glyph images into data/images/.

Design:
- Tries multiple sources (digitalthought, harappa pages, archive.org Mahadevan PDF link)
- For each NB code:
    1) query candidate page URLs (digitalthought, harappa search)
    2) parse the page for a canonical name, description, and an image URL
- Rate-limited, with retries and logging
- Produces: data/nb_signs.json and optionally data/images/NBxxx.ext

Usage:
    python scripts/scrape_indus_signs.py --download-images --out data/nb_signs.json

Notes:
- This script is conservative: it will not overwrite existing data/nb_signs.json unless
  --force is passed.
- Some sites block automated downloads; if you see repeated failures, try running with
  a human browser and/or provide local Mahadevan PDF / pre-downloaded image folder.
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import argparse
from pathlib import Path
import re
import logging
from urllib.parse import urljoin, urlparse

# --- Configuration ---
NB_MIN = 1
NB_MAX = 417

DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "images"
OUTFILE = DATA_DIR / "nb_signs.json"

# Candidate page URL patterns (editable)
CANDIDATE_URLS = [
    # digitalthought pages have /NNN.html with sign numbers from Mahadevan
    "https://digitalthought.info/{num}.html",
    # Harappa search page format (generic; scraper will try to find a matching page)
    # A direct per-sign page may not exist; we'll attempt Harappa search fallback
    "https://www.harappa.com/content/indus-sign-{num}",
    # Generic google cached view / archive may be added here by hand if needed
]

HEADERS = {
    "User-Agent": "IndusHoloSimScraper/1.0 (+https://your-repo.example) - polite research bot"
}

RATE_LIMIT_SECONDS = 1.25   # seconds between requests (polite)
RETRIES = 3
TIMEOUT = 15

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def zero_pad(nb):
    return f"NB{nb:03d}"


def try_fetch(url):
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.text
            else:
                logging.debug(f"URL {url} returned status {r.status_code}")
                return None
        except Exception as e:
            logging.debug(f"Fetch error {url} attempt {attempt}: {e}")
            time.sleep(0.8 * attempt)
    return None


def extract_from_digitalthought(html, base_url):
    """
    digitalthought page structure (observed): title contains sign id and name,
    images may be inside <img> tags. This function is heuristic and will return
    (name, image_url_or_none, description_or_none)
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) Title heuristic
    title = None
    if soup.title:
        title = soup.title.get_text().strip()

    # 2) Find first large image in content
    img = None
    # prefer images inside article
    for sel in ("article img", ".content img", "img"):
        found = soup.select(sel)
        if found:
            img = found[0].get("src")
            break

    # Normalize image URL
    if img:
        img = urljoin(base_url, img)

    # 3) Short description from first <p>
    p = soup.find("p")
    desc = p.get_text().strip() if p else None

    return title, img, desc


def extract_generic(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text().strip() if soup.title else None

    # try to find an image that looks like the glyph (smallish)
    img = None
    all_imgs = soup.find_all("img")
    # heuristic: choose the first image whose filename contains NB or 'sign' or a small width
    for im in all_imgs:
        src = im.get("src")
        if not src:
            continue
        if re.search(r"(NB0?\d+|sign|glyph|indus)", src, re.IGNORECASE):
            img = urljoin(base_url, src)
            break
    if not img and all_imgs:
        img = urljoin(base_url, all_imgs[0].get("src"))

    # description from meta or first paragraph
    desc = None
    meta_desc = soup.find("meta", {"name": "description"})
    if meta_desc and meta_desc.get("content"):
        desc = meta_desc.get("content").strip()
    else:
        p = soup.find("p")
        desc = p.get_text().strip() if p else None

    return title, img, desc


def download_image(url, dest_path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return True
        else:
            logging.warning(f"Failed image {url} status {r.status_code}")
    except Exception as e:
        logging.warning(f"Failed image download {url}: {e}")
    return False


def main(download_images=False, force=False, outfile=OUTFILE):
    DATA_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing if present and not forcing
    if outfile.exists() and not force:
        logging.info(f"{outfile} exists; load it first and update missing entries.")
        with open(outfile, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {}

    failed = []
    sources_used = {}

    for i in range(NB_MIN, NB_MAX + 1):
        code = zero_pad(i)
        # ensure baseline placeholder
        if code not in registry:
            registry[code] = {
                "code": code,
                "name": code,
                "maha_id": None,
                "parpola_id": None,
                "occurrences": None,
                "category": None,
                "default_freq": None,
                "sigma": 0.06,
                "harmonics": [[1.0, 1.0, 0.0]],
                "image_url": None,
                "notes": "Placeholder"
            }

        # Skip entries already reasonably filled to speed up
        entry = registry[code]
        if entry.get("name") and entry.get("name") != code and not force:
            logging.debug(f"{code} already has a name, skipping (use --force to overwrite)")
            continue

        # Try candidate pages
        success = False
        for pattern in CANDIDATE_URLS:
            # digitalthought uses plain numbering (1..417)
            num_for_url = str(i)  # digitalthought expects 1..417
            candidate = pattern.format(num=num_for_url)
            logging.debug(f"Trying {candidate}")
            html = try_fetch(candidate)
            # politeness
            time.sleep(RATE_LIMIT_SECONDS)
            if not html:
                continue

            # choose extractor per site
            hostname = urlparse(candidate).hostname or ""
            if "digitalthought" in hostname:
                name, img_url, desc = extract_from_digitalthought(html, candidate)
            else:
                name, img_url, desc = extract_generic(html, candidate)

            if name:
                # sanitize name: drop long site suffixes, keep leading phrase
                name_clean = re.split(r"\||\-|—", name)[0].strip()
                # sometimes titles include the NB code — remove it
                name_clean = re.sub(rf"^NB0*{i}\b[:\-\s]*", "", name_clean, flags=re.IGNORECASE)
                name_clean = name_clean.strip() or code

                entry["name"] = name_clean
                if desc and not entry.get("notes"):
                    entry["notes"] = desc[:500]
                if img_url:
                    entry["image_url"] = img_url
                registry[code] = entry
                sources_used[code] = candidate
                success = True
                logging.info(f"Found {code} -> {name_clean} @ {candidate}")
                break

        if not success:
            logging.debug(f"No direct page found for {code}; will leave placeholder")
            failed.append(code)

    # Optionally download images
    if download_images:
        logging.info("Downloading images...")
        for code, entry in registry.items():
            img_url = entry.get("image_url")
            if not img_url:
                continue
            # build file extension
            ext = Path(urlparse(img_url).path).suffix
            if not ext:
                ext = ".png"
            outpath = IMG_DIR / f"{code}{ext}"
            if outpath.exists():
                logging.debug(f"Image exists {outpath}")
                continue
            ok = download_image(img_url, outpath)
            time.sleep(RATE_LIMIT_SECONDS)
            if ok:
                entry["image_url"] = str(outpath.resolve())
                registry[code] = entry
            else:
                logging.warning(f"Failed to download image for {code}")

    # Fill missing default_freq heuristically if None (simple heuristic)
    for code, entry in registry.items():
        if entry.get("default_freq") in (None, "", 0):
            # choose a default frequency mapped to categories if known, else fallback
            cat = (entry.get("category") or "").lower()
            if "fish" in (entry.get("name") or "").lower() or "fish" in cat:
                entry["default_freq"] = 22.0
            elif "vessel" in cat or "jar" in (entry.get("name") or "").lower():
                entry["default_freq"] = 16.0
            elif "human" in cat or "person" in (entry.get("name") or "").lower():
                entry["default_freq"] = 12.0
            else:
                entry["default_freq"] = 20.0
            registry[code] = entry

    # Save outfile
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    logging.info(f"Wrote registry to {outfile}")
    if failed:
        logging.info(f"{len(failed)} entries not found automatically; list sample: {failed[:10]}")

    # Save sources log
    srcfile = DATA_DIR / "scrape_sources.json"
    with open(srcfile, "w", encoding="utf-8") as f:
        json.dump(sources_used, f, indent=2)
    logging.info(f"Wrote sources log to {srcfile}")

    return registry


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download-images", action="store_true", help="Download images to data/images/")
    p.add_argument("--force", action="store_true", help="Force overwrite existing registry entries")
    p.add_argument("--out", default=str(OUTFILE), help="Output JSON path")
    args = p.parse_args()
    main(download_images=args.download_images, force=args.force, outfile=Path(args.out))
