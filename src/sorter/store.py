"""Yang ditulis ke runs/: cache descriptor (SQLite) dan trace per run (JSONL)."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from sorter.schemas import Descriptor, Item


class Cache:
    """Descriptor per item. Kunci: path + mtime + ukuran + model. Item yang tidak berubah
    tidak dibaca model lagi; item yang berubah otomatis dibaca ulang."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS descriptors ("
            "path TEXT, mtime REAL, size INTEGER, model TEXT, data TEXT, "
            "PRIMARY KEY (path, mtime, size, model))"
        )

    def _key(self, item: Item, model: str) -> tuple:
        return (str(item.path), item.mtime.timestamp(), item.size_bytes, model)

    def get(self, item: Item, model: str) -> Descriptor | None:
        row = self.db.execute(
            "SELECT data FROM descriptors WHERE path=? AND mtime=? AND size=? AND model=?",
            self._key(item, model),
        ).fetchone()
        return Descriptor.model_validate_json(row[0]) if row else None

    def put(self, item: Item, d: Descriptor) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO descriptors VALUES (?, ?, ?, ?, ?)",
            (*self._key(item, d.model), d.model_dump_json()),
        )
        self.db.commit()


class Trace:
    """Satu baris JSON per kejadian: panggilan model, hasil, error, dan durasinya."""

    def __init__(self, path: Path) -> None:
        self.file = path.open("a", encoding="utf-8")

    def log(self, kind: str, name: str, **data) -> None:
        record = {"ts": datetime.now().isoformat(timespec="milliseconds"), "kind": kind}
        record |= {"name": name} | data
        self.file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()