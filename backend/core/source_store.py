import json
import sqlite3
from pathlib import Path
from typing import Any


class SourceStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS answer_sources "
                "(request_id TEXT PRIMARY KEY, sources_json TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    def save(self, request_id: str, sources: list[dict[str, Any]], created_at: str) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO answer_sources VALUES (?, ?, ?)",
                (request_id, json.dumps(sources), created_at),
            )

    def get(self, request_id: str) -> list[dict[str, Any]] | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT sources_json FROM answer_sources WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None
