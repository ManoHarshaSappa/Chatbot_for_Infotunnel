"""
SQLite database helper — no install needed, built into Python.
Stores all chat conversations and messages locally.

Tables:
    conversations  — one row per chat session
    messages       — one row per message (user or assistant)
"""

import sqlite3
from pathlib import Path
from datetime import datetime

# DB file lives inside the database/ folder
DB_PATH = Path(__file__).parent / "chat_history.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call every startup."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    UNIQUE NOT NULL,
            title       TEXT    DEFAULT 'New Conversation',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS messages (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id   INTEGER NOT NULL
                                REFERENCES conversations(id) ON DELETE CASCADE,
            role              TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content           TEXT NOT NULL,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def create_conversation(session_id: str, title: str = "New Conversation") -> int:
    """Create a conversation row (or return existing one's id)."""
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO conversations (session_id, title) VALUES (?, ?)",
        (session_id, title),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM conversations WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return row["id"]


def save_message(conversation_id: int, role: str, content: str):
    """Append a message to a conversation."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content),
    )
    conn.commit()
    conn.close()


def get_messages(conversation_id: int) -> list[dict]:
    """Return all messages in a conversation, oldest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_conversations() -> list[dict]:
    """Return all conversations, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, session_id, title, created_at "
        "FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_conversation_title(session_id: str, title: str):
    """Update the title of a conversation (use first user message as title)."""
    conn = get_connection()
    conn.execute(
        "UPDATE conversations SET title = ? WHERE session_id = ?",
        (title[:80], session_id),
    )
    conn.commit()
    conn.close()


def delete_conversation(session_id: str):
    """Delete a conversation and all its messages (CASCADE)."""
    conn = get_connection()
    conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
