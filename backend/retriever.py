"""
retriever.py — Loads the saved FAISS index and performs similarity search.

The vectorstore is loaded once and cached in memory for the session.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_PATH = ROOT / "data" / "faiss_index"

# Module-level cache — loaded once per process
_vectorstore: FAISS | None = None


def _load_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        if not FAISS_PATH.exists():
            raise RuntimeError(
                "FAISS index not found. Run this first:\n"
                "    python backend/ingest.py"
            )
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vectorstore = FAISS.load_local(
            str(FAISS_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def search(query: str, k: int = 5) -> list[dict]:
    """
    Search the knowledge base for the top-k most relevant chunks.

    Returns a list of dicts with keys:
        content  — the chunk text
        source   — which section it came from
        image    — image URL if available (else None)
    """
    vs = _load_vectorstore()
    docs = vs.similarity_search(query, k=k)
    results = []
    for doc in docs:
        results.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "image": doc.metadata.get("image"),
            }
        )
    return results


def search_text_only(query: str, k: int = 5) -> list[str]:
    """Convenience function — returns only the text content of top-k chunks."""
    return [r["content"] for r in search(query, k=k)]
