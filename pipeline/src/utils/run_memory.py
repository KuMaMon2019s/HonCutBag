"""Run-scoped, three-tier memory backed by SQLite."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Sequence


Embedder = Callable[[str], Sequence[float]]
Summarizer = Callable[[list[dict[str, Any]]], str]
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_VECTOR_SIZE = 256


def hashed_bag_of_words(text: str) -> list[float]:
    """Return a stable dependency-free hashed bag-of-words vector."""
    vector = [0.0] * _VECTOR_SIZE
    for token in _TOKEN_RE.findall(text.casefold()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % _VECTOR_SIZE
        vector[index] += 1.0
    return vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate cosine similarity without requiring numpy."""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _default_summarizer(entries: list[dict[str, Any]]) -> str:
    """Compress entries with the existing streaming LLM client."""
    from quality.supervision_agent import _call_llm

    payload = "\n".join(f"{entry['role']}: {entry['content']}" for entry in entries)
    prompt = (
        "Compress these run-memory entries into a concise factual summary. "
        "Preserve decisions, failures, metrics, and artifact names. Do not add facts.\n\n"
        f"{payload}"
    )
    return _call_llm(prompt, {"max_tokens": 512})


class RunMemory:
    """Store and retrieve short-term, summary, and similarity memory tiers."""

    def __init__(
        self,
        output_dir: Path,
        *,
        messages_per_summary: int = 3,
        embedder: Embedder | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        if messages_per_summary < 1:
            raise ValueError("messages_per_summary must be at least 1")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "run_memory.db"
        self.messages_per_summary = messages_per_summary
        self.embedder = embedder or hashed_bag_of_words
        self.summarizer = summarizer or _default_summarizer
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL CHECK(type IN ('message', 'summary')),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    related_ids TEXT,
                    summarized INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_tier "
                "ON memories(type, summarized, id)"
            )

    def add(self, role: str, content: str, meta: dict[str, Any] | None = None) -> int:
        """Add a raw entry and summarize the oldest full pending batch."""
        stored_content = content
        if meta:
            stored_content = f"{content}\nmeta={json.dumps(meta, ensure_ascii=False, sort_keys=True)}"
        embedding = json.dumps(list(self.embedder(stored_content)))
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memories(type, role, content, embedding) VALUES (?, ?, ?, ?)",
                ("message", role, stored_content, embedding),
            )
            entry_id = int(cursor.lastrowid)
        self._summarize_pending()
        return entry_id

    def _summarize_pending(self) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories WHERE type='message' AND summarized=0 "
                "ORDER BY id ASC LIMIT ?",
                (self.messages_per_summary,),
            ).fetchall()
        if len(rows) < self.messages_per_summary:
            return

        entries = [self._row_to_dict(row) for row in rows]
        summary = self.summarizer(entries)
        related_ids = [entry["id"] for entry in entries]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in related_ids)
            still_pending = connection.execute(
                f"SELECT COUNT(*) FROM memories WHERE summarized=0 AND id IN ({placeholders})",
                related_ids,
            ).fetchone()[0]
            if still_pending != len(related_ids):
                connection.rollback()
                return
            connection.execute(
                "INSERT INTO memories(type, role, content, embedding, related_ids) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "summary",
                    "summary",
                    summary,
                    json.dumps(list(self.embedder(summary))),
                    json.dumps(related_ids),
                ),
            )
            connection.execute(
                f"UPDATE memories SET summarized=1 WHERE id IN ({placeholders})",
                related_ids,
            )
            connection.commit()

    def get(
        self,
        query: str,
        short_term_limit: int = 5,
        summary_limit: int = 10,
        rag_limit: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return the short-term, newest-summary, and raw similarity tiers."""
        with self._connect() as connection:
            short_rows = connection.execute(
                "SELECT * FROM memories WHERE type='message' AND summarized=0 "
                "ORDER BY id DESC LIMIT ?",
                (max(0, short_term_limit),),
            ).fetchall()
            summary_rows = connection.execute(
                "SELECT * FROM memories WHERE type='summary' ORDER BY id DESC LIMIT ?",
                (max(0, summary_limit),),
            ).fetchall()
            raw_rows = connection.execute(
                "SELECT * FROM memories WHERE type='message'", ()
            ).fetchall()
        return {
            "short_term": [self._row_to_dict(row) for row in short_rows],
            "summaries": [self._row_to_dict(row) for row in summary_rows],
            "rag": self._rank(query, raw_rows, rag_limit),
        }

    def deep_retrieve(self, keyword: str, summary_limit: int = 3) -> list[dict[str, Any]]:
        """Rank summaries and expand their related raw entry IDs."""
        with self._connect() as connection:
            summaries = connection.execute(
                "SELECT * FROM memories WHERE type='summary'"
            ).fetchall()
            ranked = self._rank(keyword, summaries, summary_limit)
            related_ids: list[int] = []
            for summary in ranked:
                related_ids.extend(summary.get("related_ids") or [])
            if not related_ids:
                return []
            placeholders = ",".join("?" for _ in related_ids)
            rows = connection.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})", related_ids
            ).fetchall()
        by_id = {row["id"]: self._row_to_dict(row) for row in rows}
        return [by_id[entry_id] for entry_id in related_ids if entry_id in by_id]

    def _rank(self, query: str, rows: Sequence[sqlite3.Row], limit: int) -> list[dict[str, Any]]:
        query_vector = list(self.embedder(query))
        ranked = []
        for row in rows:
            item = self._row_to_dict(row)
            item["score"] = cosine_similarity(query_vector, item["embedding"])
            ranked.append(item)
        ranked.sort(key=lambda item: (-item["score"], -item["id"]))
        return ranked[: max(0, limit)]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["embedding"] = json.loads(item["embedding"])
        item["related_ids"] = json.loads(item["related_ids"] or "[]")
        item["summarized"] = bool(item["summarized"])
        return item


def get_run_memory(output_dir: Path, query: str, **kwargs: Any) -> dict[str, Any]:
    """Module-level retrieval helper for future phase and quality consumers."""
    return RunMemory(output_dir).get(query, **kwargs)

