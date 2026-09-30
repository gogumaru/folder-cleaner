"""Ambil cuplikan kecil dari satu item untuk dibaca model. BACA-SAJA.

Yang diambil hanya secukupnya: halaman pertama PDF, thumbnail gambar, awal teks, atau daftar
nama di level pertama folder. Ukuran file tidak berpengaruh ke waktu proses.
"""

import html
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from sorter.config import Settings
from sorter.schemas import Item, ItemKind

pymupdf.TOOLS.mupdf_display_errors(False)  # file rusak cukup jadi "name_only", tanpa spam log


@dataclass
class Snippet:
    source: str  # "text", "image", "listing", "name_only"
    text: str = ""
    image: bytes | None = None  # JPEG
    note: str = ""  # keterangan untuk prompt, misal "PDF 3 halaman, halaman 1 dirender"


def extract(item: Item, cfg: Settings) -> Snippet:
    try:
        if item.kind is ItemKind.FOLDER:
            return _listing(item, cfg)
        if item.suffix == ".pdf":
            return _pdf(item, cfg)
        if item.suffix in cfg.image_suffixes:
            return _image(item, cfg)
        if item.suffix == ".docx":
            return _docx(item, cfg)
        if item.suffix in cfg.text_suffixes:
            return _text(item, cfg)
    except Exception as exc:  # file rusak atau format tak dikenal: model tetap dapat namanya
        return Snippet("name_only", note=f"isi tidak bisa dibaca ({type(exc).__name__})")
    return Snippet("name_only", note="format ini tidak dibaca isinya")


def _pdf(item: Item, cfg: Settings) -> Snippet:
    with pymupdf.open(item.path) as doc:
        pages = doc.page_count
        text = doc[0].get_text().strip()
        if len(text) >= 40:
            return Snippet("text", text=text[: cfg.text_max_chars], note=f"PDF {pages} halaman")
        # Hampir tanpa teks: kemungkinan hasil scan, jadi halaman pertama dikirim sebagai gambar
        return Snippet(
            "image",
            image=_render(doc[0], cfg),
            note=f"PDF {pages} halaman tanpa teks, halaman 1 dirender",
        )


def _image(item: Item, cfg: Settings) -> Snippet:
    try:
       with pymupdf.open(item.path) as doc:
           return Snippet("image", image=_render(doc[0], cfg), note="thumbnail gambar")
    except Exception:
       if sys.platform != "darwin":
           raise
       # HEIC dari iPhone (atau format lain yang tidak dikenal pymupdf): pakai `sips` bawaan macOS
       return Snippet("image", image=_sips_jpeg(item.path, cfg), note="thumbnail foto")

def _sips_jpeg(path: Path, cfg: Settings) -> bytes:
   """Konversi ke JPEG kecil lewat `sips`. Hasilnya ditulis ke folder sementara sistem,
   tidak pernah ke folder yang dipindai, dan langsung dihapus setelah dibaca."""
   with tempfile.TemporaryDirectory() as tmp:
       out = Path(tmp) / "thumb.jpg"
       size = str(cfg.image_max_px)
       cmd = ["sips", "-s", "format", "jpeg", "-Z", size, str(path), "--out", str(out)]
       subprocess.run(cmd, check=True, capture_output=True, timeout=30)
       return out.read_bytes()


def _render(page: pymupdf.Page, cfg: Settings) -> bytes:
    zoom = min(1.0, cfg.image_max_px / max(page.rect.width, page.rect.height))
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("jpg")


def _docx(item: Item, cfg: Settings) -> Snippet:
    with zipfile.ZipFile(item.path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    text = html.unescape(re.sub(r"<[^>]+>", "", xml.replace("</w:p>", "\n"))).strip()
    return Snippet("text", text=text[: cfg.text_max_chars], note="dokumen Word")


def _text(item: Item, cfg: Settings) -> Snippet:
    with item.path.open("rb") as f:
        raw = f.read(cfg.text_max_chars * 4)
    text = raw.decode("utf-8", errors="replace")[: cfg.text_max_chars]
    return Snippet("text", text=text, note="awal file teks")


def _listing(item: Item, cfg: Settings) -> Snippet:
    entries = sorted(p for p in item.path.iterdir() if not p.name.startswith("."))
    dirs = sum(p.is_dir() for p in entries)
    shown = [p.name + ("/" if p.is_dir() else "") for p in entries[: cfg.folder_listing_max]]
    more = f"\n... dan {len(entries) - len(shown)} lagi" if len(entries) > len(shown) else ""
    return Snippet(
        "listing",
        text="\n".join(shown) + more,
        note=f"folder berisi {len(entries) - dirs} file dan {dirs} subfolder",
    )