"""
ingest.py — Run this ONCE before starting the app.

What it does:
  1. Loads scraped_content.json
  2. Splits text into chunks
  3. Embeds each chunk with OpenAI text-embedding-3-small
  4. Saves the FAISS index to data/faiss_index/ on disk

Run:
    python backend/ingest.py

After this, the app loads the saved index instantly — no rebuilding every restart.
"""

import json
import sys
from pathlib import Path

# Make sure project root is in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Paths
JSON_CANDIDATES = [
    ROOT / "data" / "scraped_content.json",
    ROOT / "scraped_content.json",        # legacy location
]
FAISS_PATH = ROOT / "data" / "faiss_index"


def find_json() -> Path:
    for p in JSON_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "scraped_content.json not found. "
        "Expected at data/scraped_content.json or project root."
    )


def extract_documents(json_path: Path) -> list[Document]:
    """
    Parse scraped_content.json and return a list of LangChain Documents.
    Each document carries the section key as metadata for source tracking.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for section_key, entries in data.items():
        for entry in entries:
            if isinstance(entry, str) and entry.strip():
                documents.append(
                    Document(page_content=entry.strip(), metadata={"source": section_key})
                )
            elif isinstance(entry, dict):
                if "caption" in entry and entry["caption"].strip():
                    documents.append(
                        Document(
                            page_content=entry["caption"].strip(),
                            metadata={
                                "source": section_key,
                                "image": entry.get("images", [None])[0],
                            },
                        )
                    )
    return documents


def build_and_save_index():
    print("=" * 50)
    print("DARIA 3.0 — Building Knowledge Base")
    print("=" * 50)

    json_path = find_json()
    print(f"Loading data from: {json_path}")

    raw_docs = extract_documents(json_path)
    print(f"Extracted {len(raw_docs)} raw text blocks")

    # Split into smaller chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks")

    # Embed using OpenAI (cheap + high quality)
    print("Generating embeddings with OpenAI text-embedding-3-small ...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Build FAISS index from chunks
    print("Building FAISS index ...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save to disk so the app doesn't rebuild on every restart
    FAISS_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_PATH))

    print(f"\nFAISS index saved to: {FAISS_PATH}")
    print("Setup complete. You can now run the app.")
    print("=" * 50)
    return vectorstore


if __name__ == "__main__":
    build_and_save_index()
