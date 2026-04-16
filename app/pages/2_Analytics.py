import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from backend.retriever import search
from backend.llm import get_answer

st.set_page_config(
    page_title="Analytics — DARIA 3.0",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .main .block-container {
        padding: 1.2rem 1rem 2rem !important;
        max-width: 1100px;
    }
    .page-title { font-size: 1.6rem; font-weight: 800; color: #F3F4F6; margin-bottom: 4px; }
    .page-sub   { font-size: 0.85rem; color: #6B7280; margin-bottom: 1.5rem; }
    .tag {
        display: inline-block; background: #1E1B4B; color: #A5B4FC;
        font-size: 0.72rem; font-weight: 700; padding: 3px 10px;
        border-radius: 99px; margin-bottom: 10px; letter-spacing: 0.05em;
    }
    .result-box {
        background: #161825; border: 1px solid #252840;
        border-radius: 12px; padding: 18px; height: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="page-title">📊 Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Run benchmark questions to test retrieval quality and answer accuracy</p>',
            unsafe_allow_html=True)

BENCHMARKS = [
    {"category": "Data Analysis",
     "question": "What type of waveform data does Acoustic Emission testing collect?"},
    {"category": "Contextual Understanding",
     "question": "What is the primary purpose of Acoustic Emission technology in infrastructure?"},
    {"category": "Practical Applications",
     "question": "What are the advantages and limitations of Dye Penetrant Testing?"},
    {"category": "Technical Explanation",
     "question": "How does infrared thermography detect defects in structures?"},
    {"category": "Technology Comparison",
     "question": "How does Ground Penetrating Radar differ from Acoustic Emission testing?"},
]

with st.sidebar:
    st.markdown("### 📊 Analytics")
    st.divider()
    run_all    = st.button("▶  Run All Benchmarks", type="primary", use_container_width=True)
    st.divider()
    st.markdown("#### Test Custom Question")
    custom_q   = st.text_area("", height=100, placeholder="Type any question...")
    run_custom = st.button("Run Test", use_container_width=True)
    st.divider()
    k_val = st.slider("Chunks to retrieve (k)", 1, 10, 5)

if run_all:
    st.markdown("### Benchmark Results")
    summary_rows = []
    for item in BENCHMARKS:
        retrieved = search(item["question"], k=k_val)
        chunks    = [r["content"] for r in retrieved]
        answer    = get_answer(item["question"], chunks)
        cl, cr    = st.columns(2, gap="medium")
        with cl:
            with st.container(border=True):
                st.markdown(f'<span class="tag">{item["category"]}</span>', unsafe_allow_html=True)
                st.markdown(f"**{item['question']}**")
                with st.expander(f"📄 {len(retrieved)} sources"):
                    for i, r in enumerate(retrieved, 1):
                        st.markdown(f"**{i}.** `{r['source']}`")
                        st.caption(r["content"][:200] + "...")
        with cr:
            with st.container(border=True):
                st.markdown('<span class="tag">DARIA Answer</span>', unsafe_allow_html=True)
                st.markdown(answer)
        summary_rows.append({
            "Category": item["category"],
            "Question": item["question"][:58] + "...",
            "Sources":  len(retrieved),
            "Words":    len(answer.split()),
        })
    st.divider()
    st.markdown("### Summary")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

elif run_custom and custom_q.strip():
    st.markdown("### Custom Test")
    retrieved = search(custom_q, k=k_val)
    chunks    = [r["content"] for r in retrieved]
    answer    = get_answer(custom_q, chunks)
    cl, cr    = st.columns(2, gap="medium")
    with cl:
        st.markdown("**Retrieved Sources**")
        for i, r in enumerate(retrieved, 1):
            with st.container(border=True):
                st.markdown(f"**Chunk {i}** · `{r['source']}`")
                st.caption(r["content"][:300] + ("..." if len(r["content"]) > 300 else ""))
    with cr:
        st.markdown("**DARIA's Answer**")
        with st.container(border=True):
            st.markdown(answer)
            st.caption(f"{len(retrieved)} sources · {len(answer.split())} words")
else:
    st.info("Use the sidebar to run all benchmarks or test a custom question.")
    st.markdown("**Benchmark Questions:**")
    for b in BENCHMARKS:
        st.markdown(f'<span class="tag">{b["category"]}</span> {b["question"]}<br>',
                    unsafe_allow_html=True)
