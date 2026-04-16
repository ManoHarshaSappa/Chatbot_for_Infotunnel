import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="DARIA 3.0 — InfoTunnel AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}

    body { background-color: #0D0F1A; }

    /* Remove main block padding */
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    .landing-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 0px 0px;
        text-align: center;
    }
    .logo-badge {
        background: linear-gradient(135deg, #4F8BF9 0%, #A78BFA 100%);
        color: white;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        padding: 6px 18px;
        border-radius: 99px;
        display: inline-block;
        margin-bottom: 24px;
    }
    .landing-title {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FFFFFF 0%, #A5B4FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 16px;
    }
    .landing-sub {
        font-size: 1.15rem;
        color: #9CA3AF;
        max-width: 560px;
        line-height: 1.7;
        margin-bottom: 40px;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        max-width: 820px;
        margin: 0 auto 48px;
        width: 100%;
    }
    .feature-card {
        background: #161825;
        border: 1px solid #252840;
        border-radius: 14px;
        padding: 22px 18px;
        text-align: left;
        transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: #4F8BF9; }
    .feature-icon { font-size: 1.6rem; margin-bottom: 10px; }
    .feature-title { font-size: 0.95rem; font-weight: 700; color: #F3F4F6; margin-bottom: 6px; }
    .feature-desc  { font-size: 0.8rem; color: #6B7280; line-height: 1.5; }
    .status-row {
        display: flex;
        gap: 24px;
        justify-content: center;
        margin-top: 16px;
        font-size: 0.8rem;
        color: #4B5563;
    }
    .status-dot-ok  { color: #10B981; margin-right: 4px; }
    .status-dot-err { color: #EF4444; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Landing ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="landing-wrap">
    <div class="logo-badge">FHWA · InfoTunnel · NDE</div>
    <div class="landing-title">DARIA 3.0</div>
    <div class="landing-sub">
        Domain-specific AI for Retrieval and Integrated Analysis.<br>
        Ask anything about infrastructure inspection, NDE techniques,
        bridge and tunnel testing — in plain conversation.
    </div>
</div>
""", unsafe_allow_html=True)

# ── CTA button ────────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    if st.button("Start Chatting →", type="primary", use_container_width=True):
        st.switch_page("pages/1_AI_Assistant.py")

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature cards ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">💬</div>
        <div class="feature-title">Natural Conversation</div>
        <div class="feature-desc">Chat like you'd talk to a colleague.
        DARIA remembers context and asks follow-up questions.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">RAG-Powered Answers</div>
        <div class="feature-desc">Every technical answer is grounded in
        the official FHWA InfoTunnel knowledge base — no hallucinations.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🎤</div>
        <div class="feature-title">Voice Input</div>
        <div class="feature-desc">Record your question and Whisper
        transcribes it instantly. Hands-free inspection support.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🌡️</div>
        <div class="feature-title">Infrared Thermography</div>
        <div class="feature-desc">Active, passive, and high-speed IRT
        for surface and subsurface defect detection.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🔊</div>
        <div class="feature-title">Acoustic Emission</div>
        <div class="feature-desc">AE waveform analysis, equipment, field
        procedures, and interpretation guidance.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📡</div>
        <div class="feature-title">GPR & More</div>
        <div class="feature-desc">Ground Penetrating Radar, Half-Cell
        Potential, Dye Penetrant Testing, ECA, and more.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── System status ─────────────────────────────────────────────────────────────
faiss_ok = (ROOT / "data" / "faiss_index").exists()
db_ok    = (ROOT / "database" / "chat_history.db").exists()
data_ok  = (ROOT / "data" / "scraped_content.json").exists()

def dot(ok): return '<span class="status-dot-ok">●</span>' if ok else '<span class="status-dot-err">●</span>'

st.markdown(f"""
<div class="status-row">
    {dot(faiss_ok)} Vector Index &nbsp;|&nbsp;
    {dot(data_ok)} Knowledge Base &nbsp;|&nbsp;
    {dot(db_ok)} Chat Database &nbsp;|&nbsp;
    <span>GPT-4o-mini &nbsp;·&nbsp; Whisper &nbsp;·&nbsp; FAISS</span>
</div>
""", unsafe_allow_html=True)
