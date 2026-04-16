import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from database.db import (init_db, create_conversation, save_message,
                          get_messages, update_conversation_title)
from backend.retriever import search
from backend.llm import get_answer_stream, get_suggestions, transcribe_audio

st.set_page_config(
    page_title="AI Assistant — DARIA 4.0",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: none;
    }

    /* Remove excessive padding from main content block */
    .main .block-container {
        padding: 1.2rem 1rem 2rem !important;
        max-width: 860px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0D0F1A;
        border-right: 1px solid #1E2130;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 1.5px solid #252840 !important;
        background: #161825 !important;
        font-size: 0.95rem !important;
        padding: 12px 16px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #4F8BF9 !important;
        box-shadow: 0 0 0 2px rgba(79,139,249,0.15) !important;
    }

    /* Source expander */
    [data-testid="stExpander"] {
        background: #10121E;
        border: 1px solid #1E2130;
        border-radius: 10px;
        margin-top: 2px;
    }
    details summary { font-size: 0.75rem; color: #4B5563; }

    /* Voice transcribed box */
    .voice-box {
        background: #0F172A;
        border: 1.5px solid #3B82F6;
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 0.85rem;
        color: #93C5FD;
        margin: 8px 0 6px;
        line-height: 1.55;
    }
    .voice-note {
        font-size: 0.68rem;
        color: #374151;
        margin-top: 5px;
    }

    /* Suggestion chips */
    .sugg-label {
        font-size: 0.72rem;
        color: #4B5563;
        margin: 12px 0 6px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Page header */
    .pg-title { font-size: 1.15rem; font-weight: 700; color: #E5E7EB; }
    .pg-sub   { font-size: 0.78rem; color: #4B5563; margin-bottom: 10px; }

    /* Hide audio widget error text (cosmetic browser bug) */
    [data-testid="stAudioInput"] > div > div:last-child > div:last-child {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ── DARIA welcome ─────────────────────────────────────────────────────────────
WELCOME = (
    "Hey there! 👋 I'm **DARIA 4.0** — your AI assistant for the **InfoTunnel** platform "
    "by the Federal Highway Administration (FHWA).\n\n"
    "I'm trained on NDE and infrastructure inspection data. Here's what I can help with:\n\n"
    "🔊 **Acoustic Emission (AE)** testing\n"
    "🌡️ **Infrared Thermography** — active & high-speed\n"
    "📡 **Ground Penetrating Radar (GPR)**\n"
    "⚡ **Half-Cell Potential (HCP)** testing\n"
    "🔬 **Dye Penetrant Testing (DPT)**\n"
    "⚙️ **Electrical Conductivity Array (ECA)**\n"
    "🏗️ Bridge, tunnel & steel structure inspection\n\n"
    "Go ahead — ask me anything! I'm here to chat 😊"
)

# ── Init ──────────────────────────────────────────────────────────────────────
init_db()

def _new_session():
    st.session_state.session_id       = str(uuid.uuid4())
    st.session_state.conversation_id  = create_conversation(st.session_state.session_id)
    st.session_state.voice_pending    = None
    st.session_state.voice_state      = "idle"   # "idle" | "transcribed"
    st.session_state.audio_key        = 0
    st.session_state.suggestions      = []
    st.session_state._quick_query     = None

for k, v in [
    ("session_id",      None),
    ("conversation_id", None),
    ("voice_pending",   None),
    ("voice_state",     "idle"),
    ("audio_key",       0),
    ("suggestions",     []),
    ("_quick_query",    None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.session_id is None:
    _new_session()
if st.session_state.conversation_id is None:
    st.session_state.conversation_id = create_conversation(st.session_state.session_id)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏗️ DARIA 4.0")
    st.caption("InfoTunnel · NDE Knowledge Assistant")
    st.divider()

    if st.button("＋  New Conversation", use_container_width=True, type="primary"):
        _new_session()
        st.rerun()

    st.divider()

    # ── Voice input — proper state machine ───────────────────────────────────
    st.markdown("#### 🎤 Voice Input")
    st.caption("Record → auto-transcribed by Whisper")

    # Key increments after each recording to reset the widget (also clears any error UI)
    audio_input = st.audio_input(
        "Record",
        key=f"audio_{st.session_state.audio_key}",
        label_visibility="collapsed",
    )

    # Only transcribe when state is idle and new audio arrives
    if audio_input is not None and st.session_state.voice_state == "idle":
        with st.spinner("Transcribing..."):
            try:
                text = transcribe_audio(audio_input)
                st.session_state.voice_pending = text
                st.session_state.voice_state   = "transcribed"
                st.session_state.audio_key    += 1   # ← resets widget, removes error message
            except Exception as e:
                st.error(f"Transcription failed: {e}")
        st.rerun()

    # Show transcribed text + Send / Clear buttons
    if st.session_state.voice_state == "transcribed" and st.session_state.voice_pending:
        st.markdown(
            f'<div class="voice-box">🎙️ "{st.session_state.voice_pending}"'
            f'<div class="voice-note">Transcription complete. Click Send to submit.</div></div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("Send ↑", use_container_width=True, type="primary", key="voice_send"):
                st.session_state._quick_query  = st.session_state.voice_pending
                st.session_state.voice_pending = None
                st.session_state.voice_state   = "idle"
                st.rerun()
        with c2:
            if st.button("✕", use_container_width=True, key="voice_clear"):
                st.session_state.voice_pending = None
                st.session_state.voice_state   = "idle"
                st.rerun()

    st.divider()

    # ── Quick Questions ───────────────────────────────────────────────────────
    st.markdown("#### 💡 Quick Questions")
    quick = [
        "What is Acoustic Emission testing?",
        "How does Ground Penetrating Radar work?",
        "Advantages of Infrared Thermography?",
        "What is Half-Cell Potential testing?",
        "Compare DPT and AE testing",
    ]
    for q in quick:
        label = q[:36] + ("..." if len(q) > 36 else "")
        if st.button(label, key=f"qq_{q}", use_container_width=True):
            st.session_state._quick_query = q
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<p class="pg-title">💬 AI Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="pg-sub">Ask anything about NDE techniques, infrastructure inspection, '
    'bridges and tunnels.</p>',
    unsafe_allow_html=True,
)

messages = get_messages(st.session_state.conversation_id)

# Welcome message on empty conversation
if not messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(WELCOME)

# Render history
for msg in messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ── Suggestion chips (shown after last assistant reply) ───────────────────────
if st.session_state.suggestions and not st.session_state._quick_query:
    st.markdown('<p class="sugg-label">💡 You might want to ask:</p>', unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.suggestions))
    for col, sugg in zip(cols, st.session_state.suggestions):
        with col:
            if st.button(sugg, key=f"sugg_{hash(sugg)}", use_container_width=True):
                st.session_state._quick_query  = sugg
                st.session_state.suggestions   = []
                st.rerun()

# ── Resolve query ─────────────────────────────────────────────────────────────
query = None

# Quick query (from voice Send, suggestion chip, or sidebar button)
if st.session_state._quick_query:
    query = st.session_state._quick_query
    st.session_state._quick_query = None

# Text input
text_input = st.chat_input("Ask me anything...")
if text_input:
    query = text_input
    st.session_state.suggestions = []   # clear old suggestions when user types

# ── Process & stream ──────────────────────────────────────────────────────────
if query:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)
    save_message(st.session_state.conversation_id, "user", query)

    if not messages:
        update_conversation_title(st.session_state.session_id, query)

    with st.spinner(""):
        retrieved = search(query, k=5)
        chunks    = [r["content"] for r in retrieved]

    history = get_messages(st.session_state.conversation_id)[:-1]

    # Stream response — ChatGPT typing effect
    with st.chat_message("assistant", avatar="🤖"):
        answer = st.write_stream(get_answer_stream(query, chunks, history))

        # Sources expander (technical answers only)
        if retrieved and not all(len(r["content"]) < 50 for r in retrieved):
            with st.expander("📄 Sources"):
                for i, r in enumerate(retrieved, 1):
                    st.markdown(f"**{i}.** `{r['source']}`")
                    st.caption(r["content"][:220] + ("..." if len(r["content"]) > 220 else ""))

    save_message(st.session_state.conversation_id, "assistant", answer)

    # Generate follow-up suggestions (non-blocking, after answer saved)
    sugg = get_suggestions(query, answer)
    st.session_state.suggestions = sugg

    st.rerun()
