"""Memilah setiap file ke kategori rancangan agen (M3 hybrid).

Model membaca ringkasan setiap file (hasil describe), lalu memilih satu kategori atau "Tidak
cocok". Pilihannya dikunci lewat skema JSON, jadi tidak bisa mengarang kategori. Setiap file
dipilah dua kali, kedua kalinya dengan urutan kategori dibalik. Kalau jawabannya beda atau
"Tidak cocok", file masuk daftar review dan tidak dipindah. Polanya sama dengan classify (M5).
"""

from collections.abc import Callable
from typing import Protocol

from sorter.agent import NONE
from sorter.llm import ModelError
from sorter.schemas import Category, Descriptor

PROMPT = """Pilih kategori folder untuk file ini berdasarkan ISINYA.

Kategori:
{categories}
- {none}: tidak ada kategori yang benar-benar cocok.

Pilih kategori yang deskripsinya cocok dengan file ini. Jangan memilih kategori hanya karena
file ini bukan kategori lain; kalau tidak ada yang cocok, pilih "{none}".

{file}"""


class Chooser(Protocol):
    def choose(self, prompt: str, choices: list[str]) -> str: ...


def build_prompt(d: Descriptor, categories: list[Category]) -> str:
    lines = "\n".join(f"- {c.name}: {c.description}" for c in categories)
    desc = d.description
    # Sama dengan M2: untuk gambar, nama file tidak ikut supaya tidak bias
    name = "" if d.source == "image" else f"Nama file: {d.name}\n"
    file = (f"{name}Jenis: {desc.doc_type}\nIsi: {desc.summary}\n"
            f"Kata kunci: {', '.join(desc.keywords)}")  # fmt: skip
    return PROMPT.format(categories=lines, none=NONE, file=file)


def assign(
    descriptors: list[Descriptor], categories: list[Category], model: Chooser,
    on_done: Callable[[str, str | None, str | None], None] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:  # fmt: skip
    """Mengembalikan ({nama: kategori}, {nama: alasan masuk review})."""
    flipped = categories[::-1]
    names = [c.name for c in categories] + [NONE]
    names_flipped = [c.name for c in flipped] + [NONE]
    assigned: dict[str, str] = {}
    review: dict[str, str] = {}
    for d in descriptors:
        try:
            first = model.choose(build_prompt(d, categories), names)
            second = model.choose(build_prompt(d, flipped), names_flipped)
        except ModelError as exc:
            review[d.name] = f"gagal dipilah: {exc}"
        else:
            if first == second and first != NONE:
                assigned[d.name] = first
            elif first == second:
                review[d.name] = "tidak cocok ke kategori mana pun"
            else:
                review[d.name] = f"ragu antara {first} dan {second}"
        if on_done:
            on_done(d.name, assigned.get(d.name), review.get(d.name))
    return assigned, review