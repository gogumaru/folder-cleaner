"""Klasifikasi file dengan model Apple on-device (M5): simulasi watcher.

Model Apple membaca cuplikan file sendiri (teks atau gambar, cara yang sama dengan Qwen di M2),
lalu memilih satu kategori dari taxonomy.json, atau "Tidak cocok". Pilihannya dibatasi lewat
guided generation, jadi model tidak bisa mengarang kategori baru.

Setiap file diklasifikasi dua kali, kedua kalinya dengan urutan kategori dibalik. Keyakinan yang
diakui model sendiri tidak bisa dipercaya (hampir selalu "tinggi"), jadi yang dipakai adalah
konsistensi: kalau dua jawaban berbeda, atau model menjawab "Tidak cocok" atau "rendah", file
masuk review queue dan tidak dipindah otomatis.


Evaluasi membandingkan pilihan model Apple dengan keputusan agen Qwen. Ini kesepakatan, bukan
akurasi: agen juga bisa salah, jadi setiap perbedaan perlu dinilai manusia.
"""

import asyncio
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from sorter.config import Settings
from sorter.extract import extract
from sorter.schemas import Category, Item, Verdict

NONE = "Tidak cocok"
LEVELS = ["tinggi", "sedang", "rendah"]

INSTRUCTIONS = """Kamu memilah file di folder Downloads ke salah satu kategori di bawah.
Pilih kategori yang deskripsinya paling cocok dengan ISI file, bukan dengan nama filenya.
Kalau tidak ada yang benar-benar cocok, pilih "{none}". Jangan memaksakan.

Jangan memilih kategori hanya karena file itu bukan kategori lain: kategori yang dipilih harus
cocok dengan deskripsinya sendiri. Foto biasa (pemandangan, hewan, orang) tidak otomatis masuk
kategori mana pun.

Kategori:
{categories}"""


class AppleUnavailable(Exception):
    pass


class Classifier(Protocol):
    name: str

    def choose(self, instructions: str, prompt: str, image: Path | None,
               choices: list[str]) -> dict: ...  # fmt: skip


class AppleModel:
    """Model bahasa sistem macOS lewat apple-fm-sdk. Hanya jalan di Mac dengan macOS 26+."""

    name = "apple-on-device"

    def __init__(self) -> None:
        try:
            import apple_fm_sdk as fm
        except ImportError as exc:
            raise AppleUnavailable(
                "Paket apple-fm-sdk belum terpasang. Jalankan: uv sync --extra apple"
            ) from exc
        self.fm = fm
        self.model = fm.SystemLanguageModel()
        ok, reason = self.model.is_available()
        if not ok:
            raise AppleUnavailable(f"Model Apple belum siap: {reason}")

    def choose(self, instructions: str, prompt: str, image: Path | None,
               choices: list[str]) -> dict:  # fmt: skip
        fm = self.fm

        # Kelas dibuat per panggilan karena daftar kategorinya berasal dari taxonomy.json.
        # Urutan field sengaja: alasan dulu, baru keputusan.
        @fm.generable("Keputusan kategori untuk satu file")
        class Keputusan:
            alasan: str = fm.guide("Satu kalimat: isi file ini apa, dan kenapa kategori itu")
            kategori: str = fm.guide("Kategori yang dipilih", anyOf=choices)
            keyakinan: str = fm.guide("Seberapa yakin dengan pilihan ini", anyOf=LEVELS)

        # Sesi baru untuk setiap file: konteks model kecil, riwayat file lain tidak perlu dibawa
        session = fm.LanguageModelSession(instructions=instructions, model=self.model)
        content = [prompt, fm.ImageAttachment(path=image)] if image else prompt  # Path, bukan str
        options = fm.GenerationOptions(sampling=fm.SamplingMode.greedy())
        result = asyncio.run(session.respond(content, generating=Keputusan, options=options))
        return {"category": result.kategori, "confidence": result.keyakinan,
                "reason": result.alasan}  # fmt: skip


def build_instructions(categories: list[Category]) -> str:
    lines = "\n".join(f"- {c.name}: {c.description}" for c in categories)
    return INSTRUCTIONS.format(none=NONE, categories=lines)


def build_prompt(item: Item, source: str, text: str, note: str) -> str:
    if source == "image":  # sama dengan M2: gambar dinilai tanpa nama file supaya tidak bias
        return f"File gambar ({note}). Pilih kategorinya berdasarkan apa yang terlihat."
    label = "Daftar isi folder" if source == "listing" else "Cuplikan isi"
    return f"Nama file: {item.name}\n{note}\n{label}:\n---\n{text}\n---"


def _names(categories: list[Category]) -> list[str]:
    return [c.name for c in categories] + [NONE]  # "Tidak cocok" selalu di akhir


def needs_review(first: str, second: str, confidence: str) -> bool:
    """Ragu bila dua jawaban berbeda, model sendiri menolak, atau mengaku kurang yakin."""
    return first != second or first == NONE or confidence == "rendah"


def classify_item(
    item: Item, categories: list[Category], model: Classifier, cfg: Settings,
    expected: str | None = None,
) -> Verdict:  # fmt: skip
    """Klasifikasi satu item. Baca-saja: file asli hanya dibaca untuk mengambil cuplikan."""
    t0 = time.perf_counter()
    snippet = extract(item, cfg)
    if snippet.source == "name_only":
        return Verdict(name=item.name, expected=expected, category=None, confidence="",
                       reason=f"tidak dikirim ke model: {snippet.note}", seconds=0)  # fmt: skip
    
    prompt = build_prompt(item, snippet.source, snippet.text, snippet.note)
    flipped = categories[::-1]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            image = None
            if snippet.image:  # SDK hanya menerima gambar dari path, jadi ditulis ke folder sementara
                image = Path(tmp) / "cuplikan.jpg"
                image.write_bytes(snippet.image)
            answer = model.choose(build_instructions(categories), prompt, image, _names(categories))
            again = model.choose(build_instructions(flipped), prompt, image, _names(flipped))

    except AppleUnavailable:
        raise
    except Exception as exc:  # SDK masih alpha: catat error per file, jangan hentikan semuanya
        return Verdict(name=item.name, expected=expected, category=None, confidence="",
                       reason="", seconds=round(time.perf_counter() - t0, 2),
                       error=f"{type(exc).__name__}: {exc}")  # fmt: skip
    first, second = answer["category"], again["category"]
    return Verdict(
        name=item.name, expected=expected,
        category=None if first == NONE else first,
        second=None if second == NONE else second,
        review=needs_review(first, second, answer["confidence"]),
        confidence=answer["confidence"], reason=answer["reason"],
        seconds=round(time.perf_counter() - t0, 2),
    )  # fmt: skip


def evaluate(
    items: list[Item], expected: dict[str, str], categories: list[Category],
    model: Classifier, cfg: Settings, on_done: Callable[[Verdict], None] | None = None,
) -> list[Verdict]:  # fmt: skip
    """Klasifikasi setiap item yang kategorinya dipilih agen, untuk dibandingkan."""
    results = []
    for it in items:
        if it.name not in expected:
            continue
        v = classify_item(it, categories, model, cfg, expected[it.name])
        results.append(v)
        if on_done:
            on_done(v)
    return results