# DARIA 4.0 — InfoTunnel AI Assistant

**Domain-specific AI for Retrieval and Integrated Analysis**

DARIA 4.0 is a conversational AI chatbot built for the [InfoTunnel](https://infotechnology.fhwa.dot.gov/) platform by the Federal Highway Administration (FHWA). It uses Retrieval-Augmented Generation (RAG) to answer questions about Nondestructive Evaluation (NDE) techniques for infrastructure inspection — grounded entirely in official FHWA documentation.

---

## What It Does

- Answers questions about NDE techniques (Acoustic Emission, GPR, Infrared Thermography, etc.)
- Talks like a human — casual greetings, follow-up questions, conversational flow
- Voice input via OpenAI Whisper — speak your question, get an instant answer
- Streams responses word-by-word (ChatGPT-style typing effect)
- Suggests 3 follow-up questions after every answer
- Saves every conversation to a local SQLite database
- Runs 100% locally — no cloud servers, no external databases

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | OpenAI GPT-4o-mini |
| **Voice Transcription** | OpenAI Whisper API |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector Store** | FAISS (saved to disk — no rebuild on restart) |
| **RAG Framework** | LangChain |
| **Database** | SQLite (built into Python — zero install) |
| **UI** | Streamlit 1.56+ |
| **Web Scraper** | BeautifulSoup4 + Requests |
| **Data Source** | FHWA InfoTunnel website |

---

## Architecture

```
User (Text or Voice)
        │
        ▼
[Voice] OpenAI Whisper API ──► Text
        │
        ▼
  OpenAI text-embedding-3-small
  converts question to vector
        │
        ▼
  FAISS similarity search
  on saved index (data/faiss_index/)
        │
        ▼
  Top 5 relevant chunks retrieved
  from scraped_content.json
        │
        ▼
  GPT-4o-mini generates answer
  (streamed token by token)
        │
        ▼
  Answer + 3 follow-up suggestions
  displayed in Streamlit UI
        │
        ▼
  Conversation saved to
  SQLite (database/chat_history.db)
```

---

## Project Structure

```
Chatbot_for_Infotunnel/
│
├── .env                          ← OpenAI API key (never commit this)
├── .gitignore                    ← protects .env, DB, and FAISS index
├── .streamlit/
│   └── config.toml               ← runs on port 8502
├── requirements.txt
├── setup.py                      ← one-time setup script
│
├── app/                          ← Streamlit UI
│   ├── main.py                   ← landing page
│   └── pages/
│       ├── 1_AI_Assistant.py     ← main chat interface
│       ├── 2_Analytics.py        ← benchmark evaluation
│       └── 3_Knowledge_Base.py   ← DB and index stats
│
├── backend/
│   ├── ingest.py                 ← builds FAISS index from JSON
│   ├── retriever.py              ← loads FAISS, runs similarity search
│   └── llm.py                    ← GPT-4o-mini + Whisper + suggestions
│
├── database/
│   └── db.py                     ← SQLite helpers (conversations + messages)
│
├── scraper/
│   └── scraper.py                ← scrapes FHWA InfoTunnel site
│
├── data/
│   ├── scraped_content.json      ← knowledge base (scraped FHWA data)
│   └── faiss_index/              ← vector index (built by setup.py)
│
└── archive/                      ← original DARIA 2.0 / 3.0 legacy code
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher (Anaconda recommended)
- OpenAI API key — get one at [platform.openai.com](https://platform.openai.com/api-keys)

### Step 1 — Clone the repo

```bash
git clone https://github.com/ManoHarshaSappa/Chatbot_for_Infotunnel.git
cd Chatbot_for_Infotunnel
```

### Step 2 — Add your OpenAI API key

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

If you want to run the legacy models inside `archive/Daria-3o/`, also add:

```bash
echo "HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here" >> .env
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run one-time setup

This builds the FAISS vector index from the knowledge base and initializes SQLite:

```bash
python setup.py
```

You will see:
```
Extracted 825 raw text blocks
Split into 899 chunks
Generating embeddings with OpenAI text-embedding-3-small ...
FAISS index saved to data/faiss_index/
SQLite database ready.
Setup complete!
```

### Step 5 — Launch the app

```bash
streamlit run app/main.py
```

Open your browser at **`http://localhost:8502`**

---

## How to Use

### Chat
1. Go to **AI Assistant** from the sidebar or home page
2. Type your question in the input bar at the bottom
3. DARIA responds with a streaming answer (types out word by word)
4. Click any of the 3 **follow-up suggestion chips** to continue the conversation
5. Or record your voice using the **🎤 Voice Input** in the sidebar

### Voice Input
1. Click the microphone widget in the sidebar
2. Record your question (up to ~30 seconds)
3. Whisper transcribes it automatically
4. Review the transcribed text, then click **Send ↑**

### Quick Questions
Click any preset question in the sidebar to test instantly without typing.

### Analytics
Run the 5 benchmark questions to evaluate retrieval quality and answer accuracy.

### Knowledge Base / Stats
View FAISS index status, number of indexed chunks, SQLite conversation history, and source data overview.

---

## Refreshing the Knowledge Base

If the FHWA InfoTunnel website updates and you want fresh data:

```bash
# Re-scrape the website
python scraper/scraper.py

# Rebuild the FAISS index with new data
python backend/ingest.py
```

---

## What DARIA Knows

All knowledge comes from the FHWA InfoTunnel documentation:

| NDE Technique | Description |
|--------------|-------------|
| **Acoustic Emission (AE)** | Detects stress waves from crack growth and material defects |
| **Active Infrared Thermography** | Uses heat source to reveal subsurface defects |
| **High-Speed IRT** | Fast thermographic scanning for large structures |
| **Ground Penetrating Radar (GPR)** | Subsurface imaging using radar pulses |
| **Half-Cell Potential (HCP)** | Detects corrosion probability in reinforced concrete |
| **Dye Penetrant Testing (DPT)** | Surface crack detection using liquid penetrant |
| **Electrical Conductivity Array (ECA)** | Electromagnetic testing for surface/near-surface defects |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key — used for GPT-4o-mini, Whisper, and embeddings |
| `HUGGINGFACEHUB_API_TOKEN` | Optional for legacy Hugging Face-based code in `archive/Daria-3o/` |

---

## Key Design Decisions

**Why RAG instead of fine-tuning?**
RAG retrieves actual source text at query time, so answers are always grounded in real FHWA documentation. Fine-tuning bakes knowledge into weights, which goes stale and can hallucinate.

**Why FAISS + disk persistence?**
FAISS is extremely fast for similarity search. Saving to disk means the index loads instantly on restart — no rebuilding the 899-chunk index every time.

**Why SQLite?**
Zero installation, zero configuration, built into Python. Perfect for a local-first app. Easily migrated to PostgreSQL for production.

**Why GPT-4o-mini?**
Fast, cheap, and smart enough for domain-specific Q&A. ~10x cheaper than GPT-4o with minimal quality loss for NDE topics.

---

## Academic Context

Originally developed as **DARIA 3.0** for the AIT526 course at George Mason University (Team 14, advised by Dr. Lindi Liao). **DARIA 4.0** is the current version, rebuilding the stack with OpenAI APIs, persistent storage, and a production-quality UI.

---

## License

This project is for academic and research purposes. Data sourced from the Federal Highway Administration (FHWA) InfoTunnel public website.
