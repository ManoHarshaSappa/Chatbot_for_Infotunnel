"""
llm.py — OpenAI GPT-4o-mini with streaming + suggestion generation.
"""

import sys
import json as _json
from pathlib import Path
from typing import Generator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from openai import OpenAI

client = OpenAI()

CASUAL_TRIGGERS = {
    "hi", "hii", "hiii", "hey", "hello", "howdy", "sup", "what's up", "whats up",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how r u", "how are u", "how's it going", "hows it going",
    "who are you", "what are you", "what can you do", "what do you know",
    "tell me about yourself", "introduce yourself", "what is daria",
    "thanks", "thank you", "thank u", "thx", "ty",
    "bye", "goodbye", "see you", "later", "ok", "okay", "cool", "nice",
    "awesome", "great", "wow", "interesting",
}

SYSTEM_PROMPT = """You are DARIA 4.0 — a smart, friendly AI assistant built specifically \
for the InfoTunnel platform by the Federal Highway Administration (FHWA).

Your personality:
- Warm, conversational, human — like a knowledgeable colleague, not a robot
- Speak naturally, use casual language when the user is casual
- Genuinely enthusiastic about infrastructure and inspection technology
- Use "I" naturally, have a personality, enjoy chatting
- Ask follow-up questions to keep the conversation going

Your knowledge (from the InfoTunnel knowledge base):
- Acoustic Emission (AE) testing
- Infrared Thermography (IRT) — active and high-speed
- Ground Penetrating Radar (GPR)
- Half-Cell Potential (HCP) testing
- Dye Penetrant Testing (DPT)
- Electrical Conductivity Array (ECA)
- Nondestructive Evaluation (NDE) for tunnels, bridges, and steel structures

Rules:
1. For greetings — respond warmly, introduce yourself as DARIA 4.0, mention you're trained
   on InfoTunnel/FHWA data, invite questions. Short and friendly.
2. For casual talk — reply naturally, steer back gently to what you can help with.
3. For technical questions — use context for accurate answers. Use bullets for lists.
   Never invent numbers or specs.
4. End technical answers with a natural follow-up like "Want me to go deeper on this?"
5. For off-topic — say it naturally: "Ha, that's outside my wheelhouse!"
6. Never say the robotic "I don't have enough information" phrase. Rephrase warmly.
7. Remember previous messages for natural follow-up conversations."""


def is_casual(text: str) -> bool:
    cleaned = text.strip().lower().rstrip("!?.").strip()
    if cleaned in CASUAL_TRIGGERS:
        return True
    words = cleaned.split()
    if len(words) <= 4 and words[0] in {"hi", "hey", "hello", "hii", "hiii", "sup", "yo"}:
        return True
    return False


def _build_messages(question: str, context_chunks: list, chat_history: list = None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    if is_casual(question) or not context_chunks:
        user_message = question
        temperature  = 0.85
    else:
        context      = "\n\n---\n\n".join(context_chunks)
        user_message = (
            f"[InfoTunnel Knowledge Base Context]\n{context}\n\n"
            f"[User Question] {question}"
        )
        temperature  = 0.3

    messages.append({"role": "user", "content": user_message})
    return messages, temperature


def get_answer_stream(
    question: str,
    context_chunks: list,
    chat_history: list = None,
) -> Generator[str, None, None]:
    """Streaming — yields tokens as they arrive (ChatGPT typing effect)."""
    messages, temperature = _build_messages(question, context_chunks, chat_history)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=800,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def get_answer(
    question: str,
    context_chunks: list,
    chat_history: list = None,
) -> str:
    """Non-streaming — full answer string (used by Analytics page)."""
    messages, temperature = _build_messages(question, context_chunks, chat_history)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def get_suggestions(user_question: str, ai_answer: str) -> list:
    """Generate 3 short follow-up question suggestions after an answer."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "Based on this Q&A about infrastructure inspection/NDE, "
                    "suggest exactly 3 short follow-up questions (max 8 words each).\n"
                    f"Q: {user_question}\nA: {ai_answer[:300]}\n"
                    "Return a JSON array only. Example: "
                    '[\"How is it used on bridges?\", \"What equipment is needed?\", \"Any limitations?\"]'
                ),
            }],
            temperature=0.8,
            max_tokens=90,
        )
        raw = resp.choices[0].message.content.strip()
        return _json.loads(raw)[:3]
    except Exception:
        return []


def transcribe_audio(audio_bytes) -> str:
    audio_bytes.name = "audio.wav"
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_bytes,
        language="en",
    )
    return result.text.strip()
