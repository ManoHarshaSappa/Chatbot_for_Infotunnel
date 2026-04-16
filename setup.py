"""
setup.py — First-time setup script for DARIA 3.0

Run this once before launching the app:
    python setup.py

What it does:
    1. Copies scraped_content.json to data/ folder (if needed)
    2. Builds the FAISS vector index
    3. Initializes the SQLite database
"""

import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    print("=" * 55)
    print("  DARIA 3.0 — First-Time Setup")
    print("=" * 55)

    # Step 1: Make sure scraped_content.json is in data/
    src = ROOT / "scraped_content.json"
    dst = ROOT / "data" / "scraped_content.json"
    (ROOT / "data").mkdir(exist_ok=True)

    if not dst.exists():
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied scraped_content.json → data/")
        else:
            print("scraped_content.json not found!")
            print("Run the scraper first: python scraper/scraper.py")
            sys.exit(1)
    else:
        print("scraped_content.json already in data/ — skipping copy")

    # Step 2: Build FAISS index
    print("\nBuilding FAISS index...")
    from backend.ingest import build_and_save_index
    build_and_save_index()

    # Step 3: Initialize SQLite DB
    print("\nInitializing SQLite database...")
    from database.db import init_db
    init_db()
    print("SQLite database ready at: database/chat_history.db")

    print("\n" + "=" * 55)
    print("  Setup complete!")
    print("  Start the app with:")
    print("  streamlit run app/main.py")
    print("  Then open: http://localhost:3000")
    print("=" * 55)


if __name__ == "__main__":
    main()
