"""Bentuk data inventory.json. Tipe untuk milestone berikutnya ditambah saat dibutuhkan."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ItemKind(StrEnum):
    FILE = "file"
    FOLDER = "folder"
    BUNDLE = "bundle"  # .app dan paket macOS lain: satu unit, isinya tidak dibaca
    SYMLINK = "symlink"


class Triage(StrEnum):
    APP = "app"
    INSTALLER = "installer"
    ARCHIVE = "archive"
    PROJECT = "project"
    DATASET = "dataset"
    NEEDS_MODEL = "needs_model"  # tidak jelas dari aturan, nanti dibaca model (M2)
    SKIPPED = "skipped"


class Item(BaseModel):
    path: Path
    name: str
    kind: ItemKind
    suffix: str
    size_bytes: int  # 0 untuk folder
    mtime: datetime
    triage: Triage
    reason: str
    duplicate_of: Path | None = None


class DuplicateGroup(BaseModel):
    """Hanya saran. Penghapusan selalu dilakukan user."""

    original: Path
    duplicates: list[Path]
    confirmed: bool  # True bila isinya terbukti identik
    note: str
    size_bytes: int = 0

class Description(BaseModel):
    """Jawaban yang diminta dari model. Skema ini juga dikirim ke Ollama sebagai format JSON."""

    summary: str  # 1 sampai 2 kalimat tentang isi dan kegunaannya
    doc_type: str  # label pendek, misal "struk belanja", "tiket pesawat"
    keywords: list[str]
    language: str


class Descriptor(BaseModel):
    """Satu item yang sudah dibaca model. Disimpan di cache dan descriptors.json."""

    path: Path
    name: str
    source: str  # "text", "image", "listing", atau "name_only": apa yang dilihat model
    model: str
    seconds: float
    description: Description | None = None
    error: str | None = None
    from_cache: bool = False
    duplicate_of: Path | None = None  # salinan identik: ringkasan diambil dari file aslinya



class Inventory(BaseModel):
    run_id: str
    folder: Path
    scanned_at: datetime
    content_checked: bool
    items: list[Item]
    duplicates: list[DuplicateGroup]


class Cluster(BaseModel):
    id: int
    members: list[str]  # nama item


class Category(BaseModel):
    name: str  # nama folder
    description: str  # ciri konkret, dipakai model lain untuk mengklasifikasi file baru
    tier: Literal["important", "temporary"]  # important -> ~/Documents, temporary -> ~/Downloads
    items: list[str] = []
    source: str = "agent"  # "agent" atau "triage" (kategori tetap dari aturan M1)


class Taxonomy(BaseModel):
    run_id: str
    folder: Path
    model: str
    categories: list[Category]
    unread: list[str]  # item yang isinya tidak terbaca, belum masuk kategori mana pun
    agent_steps: int
    agent_finished: bool