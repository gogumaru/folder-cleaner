"""Describe: pilih sampel, ambil cuplikan, minta model menjelaskan isinya, simpan ke cache."""

import time
from collections import defaultdict
from collections.abc import Callable
from itertools import zip_longest
from pathlib import Path
from typing import Protocol

from sorter.config import Settings
from sorter.duplicates import find_duplicates
from sorter.extract import Snippet, extract
from sorter.llm import ModelError
from sorter.schemas import Description, Descriptor, Item, Triage
from sorter.store import Cache, Trace

PROMPT = """Kamu membantu merapikan folder Downloads. Jelaskan item berikut berdasarkan ISINYA,
bukan hanya dari namanya. Nama file sering tidak informatif.

Nama: {name}
Jenis: {kind}{suffix}
Keterangan: {note}
{body}
{answer}"""

# Untuk gambar, nama file sengaja TIDAK dikirim: model kecil cenderung percaya nama
# (misal meme bernama "invoice-maret.jpg" dijelaskan sebagai faktur).
IMAGE_PROMPT = """Kamu membantu merapikan folder Downloads. Lihat gambar terlampir dan jelaskan
HANYA apa yang benar-benar terlihat: objek, orang, tempat, tulisan, dan jenis gambarnya
(foto, meme, screenshot, dokumen yang difoto, ilustrasi). Jangan mengarang hal yang tidak terlihat.

Keterangan: {note}

{answer}"""

ANSWER = """Jawab dalam JSON:
- summary: 1 sampai 2 kalimat bahasa Indonesia tentang isi dan kegunaannya
- doc_type: label pendek bahasa Indonesia, misal "struk belanja", "tiket pesawat",
  "invoice", "screenshot", "foto", "meme", "tugas kuliah"
- keywords: 3 sampai 6 kata kunci
- language: bahasa isi item (kode ISO, misal "id", "en"), atau "-" bila tidak ada teks"""

# Item yang isinya tidak bisa dibaca tidak dikirim ke model: tanpa isi, model hanya mengarang.
NOT_READ = Description(
    summary="Isi tidak dibaca; hanya nama dan akhiran file yang diketahui.",
    doc_type="tidak dibaca",
    keywords=[],
    language="-",
)


class Model(Protocol):
    name: str

    def describe(self, prompt: str, image: bytes | None = None) -> Description: ...


def pick_sample(items: list[Item], n: int) -> list[Item]:
    """Sampel merata per jenis dan per kuartal, supaya tidak didominasi satu jenis file."""
    groups: dict[tuple, list[Item]] = defaultdict(list)
    for it in items:
        if it.triage is Triage.NEEDS_MODEL:
            kind = "folder" if it.kind.value == "folder" else (it.suffix or "tanpa-akhiran")
            quarter = (it.mtime.year, (it.mtime.month - 1) // 3)
            groups[(kind, quarter)].append(it)
    ordered = [sorted(g, key=lambda it: it.name.lower()) for _, g in sorted(groups.items())]
    round_robin = [it for batch in zip_longest(*ordered) for it in batch if it]
    return round_robin[:n]


def build_prompt(item: Item, snippet: Snippet) -> str:
    if snippet.source == "image":
        return IMAGE_PROMPT.format(note=snippet.note, answer=ANSWER)
    label = "Daftar isi folder" if snippet.source == "listing" else "Cuplikan isi"
    body = f"{label}:\n---\n{snippet.text}\n---\n"
    suffix = f" ({item.suffix})" if item.suffix else ""
    return PROMPT.format(
        name=item.name, kind=item.kind.value, suffix=suffix, note=snippet.note, body=body,
        answer=ANSWER,
    )  # fmt: skip

def sample_with_copies(
    items: list[Item], n: int, cfg: Settings
) -> tuple[list[Item], dict[Path, Path]]:
    """Sampel dari file asli saja, lalu salinan identik dari file terpilih ikut ditambahkan.

    Mengembalikan (item yang diproses, {salinan: asli}). Salinan tidak dikirim ke model.
    """
    groups = find_duplicates(items, cfg)
    copies = {d: g.original for g in groups if g.confirmed for d in g.duplicates}
    chosen = pick_sample([it for it in items if it.path not in copies], n)
    picked = {it.path for it in chosen}
    chosen += [it for it in items if copies.get(it.path) in picked]
    return chosen, copies

def describe_items(
    items: list[Item],
    cfg: Settings,
    model: Model,
    cache: Cache,
    trace: Trace,
    on_done: Callable[[Descriptor], None] | None = None,
    copies: dict[Path, Path] | None = None,
) -> list[Descriptor]:
    
    copies = copies or {}
    done: dict[Path, Descriptor] = {}
    # File asli diproses dulu, supaya hasilnya siap saat salinannya tiba
    ordered = sorted(items, key=lambda it: it.path in copies)
    results = []

    for item in ordered:
        original = copies.get(item.path)
        if original in done:
            d = done[original].model_copy(
                update={"path": item.path, "name": item.name, "seconds": 0.0,
                        "from_cache": False, "duplicate_of": original}
            )  # fmt: skip
            trace.log("copy", item.name, original=original)
        elif (cached := cache.get(item, model.name)) and cached.source != "name_only":
            d = cached.model_copy(update={"from_cache": True})
            trace.log("cache_hit", item.name)
        else:
            d = _describe_one(item, cfg, model, trace)
            # Yang gagal atau tidak terbaca tidak di-cache: dicoba lagi di run berikutnya,
            # misalnya setelah format filenya didukung. Membaca ulang namanya tidak memanggil model.
            if d.description and d.source != "name_only":
                cache.put(item, d)
        done[item.path] = d
        results.append(d)
        if on_done:
            on_done(d)
    return results


def _describe_one(item: Item, cfg: Settings, model: Model, trace: Trace) -> Descriptor:
    t0 = time.perf_counter()
    snippet = extract(item, cfg)
    t_extract = time.perf_counter() - t0
    if snippet.source == "name_only":
        trace.log("not_read", item.name, note=snippet.note)
        return Descriptor(
            path=item.path, name=item.name, source=snippet.source, model=model.name,
            seconds=round(t_extract, 2), description=NOT_READ,
        )  # fmt: skip
    prompt = build_prompt(item, snippet)
    description, error = None, None
    t1 = time.perf_counter()
    try:
        description = model.describe(prompt, snippet.image)
    except ModelError as exc:
        error = str(exc)
    t_model = time.perf_counter() - t1
    trace.log(
        "model",
        item.name,
        model=model.name,
        source=snippet.source,
        prompt=prompt,
        image_bytes=len(snippet.image or b""),
        output=description.model_dump() if description else None,
        error=error,
        extract_s=round(t_extract, 3),
        model_s=round(t_model, 3),
    )
    return Descriptor(
        path=item.path,
        name=item.name,
        source=snippet.source,
        model=model.name,
        seconds=round(t_extract + t_model, 2),
        description=description,
        error=error,
    )

# ---------- Tampilan describe ----------


def _describe_line(d: Descriptor) -> str:
    if d.error:
        return f"[red]gagal[/red]  {d.name}  [dim]{d.error}[/dim]"
    doc_type = d.description.doc_type if d.description else ""
    if d.from_cache:
        return f"[dim]cache  {d.name}  {doc_type}[/dim]"
    return (
        f"[green]ok[/green]     {d.name}  [cyan]{doc_type}[/cyan]  [dim]{d.seconds:.1f} dtk[/dim]"
    )


def _print_describe(results: list[Descriptor]) -> None:
    table = Table(title="\nHasil", title_justify="left", title_style="bold")
    table.add_column("Nama", overflow="fold")
    table.add_column("Dilihat", style="dim")
    table.add_column("Jenis", style="cyan")
    table.add_column("Ringkasan", overflow="fold")
    for d in results:
        if d.description:
            table.add_row(d.name, d.source, d.description.doc_type, d.description.summary)
        else:
            table.add_row(d.name, d.source, "[red]gagal[/red]", f"[dim]{d.error}[/dim]")
    console.print(table)

    new = [d for d in results if not d.from_cache and not d.error]
    cached = sum(d.from_cache for d in results)
    failed = sum(bool(d.error) for d in results)
    console.print(
        f"[green]{len(new)}[/green] baru  ·  [dim]{cached} dari cache[/dim]  ·  "
        f"[red]{failed}[/red] gagal"
    )
    if new:
        by_source = defaultdict(list)
        for d in new:
            by_source[d.source].append(d.seconds)
        per_source = ", ".join(f"{s} {mean(v):.1f}" for s, v in sorted(by_source.items()))
        avg = mean(d.seconds for d in new)
        minutes = avg * SETTINGS.sample_size / 60
        color = "green" if minutes < 30 else "yellow"
        console.print(
            f"Rata-rata [bold]{avg:.1f} detik/item[/bold] [dim]({per_source})[/dim]\n"
            f"Perkiraan {SETTINGS.sample_size} item: [{color}]{minutes:.0f} menit[/{color}] "
            "[dim](target PRD di bawah 30 menit)[/dim]"
        )