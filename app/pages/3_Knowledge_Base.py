import sys
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from database.db import get_all_conversations, get_messages

st.set_page_config(
    page_title="Knowledge Base — DARIA 3.0",
    page_icon="🗄️",
    layout="wide",
)

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .main .block-container {
        padding: 1.2rem 1rem 2rem !important;
        max-width: 1000px;
    }
    .page-title { font-size: 1.6rem; font-weight: 800; color: #F3F4F6; margin-bottom: 4px; }
    .page-sub   { font-size: 0.85rem; color: #6B7280; margin-bottom: 1.5rem; }
    .kpi-card {
        background: #161825; border: 1px solid #252840;
        border-radius: 12px; padding: 20px 18px; text-align: center;
    }
    .kpi-val   { font-size: 2.2rem; font-weight: 800; color: #4F8BF9; }
    .kpi-label { font-size: 0.78rem; color: #6B7280; margin-top: 4px; }
    .section   { font-size: 1rem; font-weight: 700; color: #E5E7EB;
                 margin: 1.8rem 0 0.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="page-title">🗄️ Knowledge Base</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">FAISS vector index, source data, and SQLite conversation database</p>',
            unsafe_allow_html=True)

def kpi(val, label):
    return (f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-label">{label}</div></div>')

# ── FAISS ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="section">Vector Index (FAISS)</p>', unsafe_allow_html=True)
faiss_path = ROOT / "data" / "faiss_index"
if faiss_path.exists():
    files    = [f for f in faiss_path.iterdir() if f.is_file()]
    total_kb = sum(f.stat().st_size for f in files) / 1024
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi(len(files),        "Index files"),      unsafe_allow_html=True)
    c2.markdown(kpi(f"{total_kb:.0f} KB", "Size on disk"),  unsafe_allow_html=True)
    c3.markdown(kpi("text-embedding-3-small", "Model"),     unsafe_allow_html=True)
    st.success("FAISS index is ready and loaded")
else:
    st.error("FAISS index not found — run: `python setup.py`")

# ── Source data ───────────────────────────────────────────────────────────────
st.markdown('<p class="section">Source Data</p>', unsafe_allow_html=True)
json_path = ROOT / "data" / "scraped_content.json"
if json_path.exists():
    size_kb = json_path.stat().st_size / 1024
    with open(json_path) as f:
        data = json.load(f)
    sections = len(data)
    texts    = sum(1 for v in data.values() for e in v if isinstance(e, str))
    imgs     = sum(1 for v in data.values() for e in v
                   if isinstance(e, dict) and "images" in e)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi(f"{size_kb:.0f} KB", "File size"),  unsafe_allow_html=True)
    c2.markdown(kpi(sections,            "Sections"),   unsafe_allow_html=True)
    c3.markdown(kpi(texts,               "Text blocks"),unsafe_allow_html=True)
    c4.markdown(kpi(imgs,                "Images"),     unsafe_allow_html=True)
else:
    st.warning("scraped_content.json not found in data/")

# ── SQLite ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="section">Chat Database (SQLite)</p>', unsafe_allow_html=True)
db_path = ROOT / "database" / "chat_history.db"
if db_path.exists():
    db_kb  = db_path.stat().st_size / 1024
    conn   = sqlite3.connect(str(db_path))
    n_conv = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    n_msg  = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    n_user = conn.execute("SELECT COUNT(*) FROM messages WHERE role='user'").fetchone()[0]
    conn.close()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi(f"{db_kb:.1f} KB",  "DB size"),          unsafe_allow_html=True)
    c2.markdown(kpi(n_conv,             "Conversations"),     unsafe_allow_html=True)
    c3.markdown(kpi(n_msg,              "Total messages"),    unsafe_allow_html=True)
    c4.markdown(kpi(n_user,             "Questions asked"),   unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Saved Conversations**")
    convos = get_all_conversations()
    if convos:
        for c in convos[:8]:
            with st.expander(f"{c['title']}  ·  {c['created_at'][:16]}"):
                for m in get_messages(c["id"]):
                    icon = "🧑" if m["role"] == "user" else "🤖"
                    st.markdown(f"{icon} **{m['role'].title()}**: {m['content'][:300]}")
    else:
        st.caption("No conversations saved yet.")
else:
    st.info("SQLite DB will be created automatically on your first chat.")
