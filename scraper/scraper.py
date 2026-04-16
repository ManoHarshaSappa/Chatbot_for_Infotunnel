"""
scraper.py — Scrapes NDE content from the InfoTunnel FHWA website.

Run this to refresh the knowledge base with the latest content:
    python scraper/scraper.py

Output: data/scraped_content.json (same format as the original)
"""

import json
import time
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "scraped_content.json"

BASE_URL = "https://infotechnology.fhwa.dot.gov"

# Known NDE technology pages on InfoTunnel
TARGET_URLS = [
    "https://infotechnology.fhwa.dot.gov/acoustic-emission/",
    "https://infotechnology.fhwa.dot.gov/active-infrared-thermography/",
    "https://infotechnology.fhwa.dot.gov/dye-penetrant-testing/",
    "https://infotechnology.fhwa.dot.gov/electrical-conductivity-array/",
    "https://infotechnology.fhwa.dot.gov/ground-penetrating-radar/",
    "https://infotechnology.fhwa.dot.gov/half-cell-potential/",
    "https://infotechnology.fhwa.dot.gov/high-speed-infrared-thermography/",
    "https://infotechnology.fhwa.dot.gov/impact-echo/",
    "https://infotechnology.fhwa.dot.gov/magnetic-flux-leakage/",
    "https://infotechnology.fhwa.dot.gov/ultrasonic-testing/",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_page(url: str) -> dict:
    """
    Scrape a single InfoTunnel page and return structured content.
    Returns a dict: { "div_1": [...], "div_2": [...], ... }
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR fetching {url}: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    page_data = {}
    div_counter = 1

    # Target content areas: article body, main content divs
    content_area = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_="entry-content")
        or soup.find("div", id="content")
        or soup.body
    )

    if not content_area:
        return {}

    # Walk through all divs and paragraphs in content area
    for element in content_area.find_all(["div", "section", "p"], recursive=False):
        section_content = []

        # Extract paragraphs
        for p in element.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if text and len(text) > 30:  # skip very short fragments
                section_content.append(text)

        # Extract headings
        for h in element.find_all(["h1", "h2", "h3", "h4"]):
            text = h.get_text(strip=True)
            if text:
                section_content.append(text)

        # Extract list items
        for li in element.find_all("li"):
            text = li.get_text(separator=" ", strip=True)
            if text and len(text) > 20:
                section_content.append(text)

        # Extract images with captions
        for fig in element.find_all(["figure", "img"]):
            img_tag = fig.find("img") if fig.name == "figure" else fig
            caption_tag = fig.find("figcaption") if fig.name == "figure" else None

            if img_tag and img_tag.get("src"):
                img_url = urljoin(BASE_URL, img_tag["src"])
                caption = (
                    caption_tag.get_text(strip=True)
                    if caption_tag
                    else img_tag.get("alt", "")
                )
                if caption:
                    section_content.append(
                        {"caption": caption, "images": [img_url]}
                    )

        if section_content:
            key = f"div_{div_counter}"
            page_data[key] = section_content
            div_counter += 1

    return page_data


def scrape_all():
    print("=" * 50)
    print("DARIA 4.0 — Scraping InfoTunnel FHWA")
    print("=" * 50)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_data = {}
    global_counter = 1

    for url in TARGET_URLS:
        print(f"Scraping: {url}")
        page_data = scrape_page(url)

        if page_data:
            # Re-key with global counter to avoid collisions across pages
            for local_key, content in page_data.items():
                all_data[f"div_{global_counter}"] = content
                global_counter += 1
            print(f"  Extracted {len(page_data)} sections")
        else:
            print(f"  No content found (page may have changed or be unavailable)")

        time.sleep(1)  # polite crawl delay

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_data)} sections to: {OUTPUT_PATH}")
    print("Now run: python backend/ingest.py")
    print("=" * 50)


if __name__ == "__main__":
    scrape_all()
