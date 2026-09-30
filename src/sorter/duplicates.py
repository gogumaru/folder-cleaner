"""Deteksi duplikat. Hanya menandai dan menyarankan; tidak pernah menghapus apa pun.

Dua sumber bukti:
- Pola nama ("nama-2", "nama copy", "nama (1)") yang aslinya juga ada. Dikonfirmasi lewat isi:
  file dengan hash identik, folder dengan daftar isi (nama + ukuran) identik. Bila isinya
  berbeda, tetap dilaporkan tapi confirmed=False.
- File dengan isi identik walau namanya tidak mirip.

Konfirmasi hanya MEMBACA file. check_content=False membuatnya sama sekali tidak membuka isi.
"""

import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

from sorter.config import Settings
from sorter.schemas import DuplicateGroup, Item, ItemKind, Triage


def name_base(item: Item, cfg: Settings) -> str | None:
    """Nama asli yang diduga: "Innowork-24-2" -> "Innowork-24", "a copy.pdf" -> "a.pdf"."""
    n = len(item.suffix)
    stem, suffix = (item.name[:-n], item.name[-n:]) if n else (item.name, "")
    for pattern in cfg.duplicate_name_patterns:
        if m := re.match(pattern, stem):
            return m.group("base") + suffix
    return None


def file_hash(path: Path, limit: int | None = None) -> str:
    with path.open("rb") as f:
        if limit:
            return hashlib.sha256(f.read(limit)).hexdigest()
        return hashlib.file_digest(f, "sha256").hexdigest()


def folder_signature(path: Path, cfg: Settings) -> str:
    """Hash dari daftar (path relatif, ukuran) semua file di dalam folder. Bukan isi byte."""
    entries = []
    for root, _, files in os.walk(path, followlinks=False):
        for name in files:
            full = os.path.join(root, name)
            entries.append(f"{os.path.relpath(full, path)}\t{os.lstat(full).st_size}")
        if len(entries) >= cfg.folder_signature_limit:
            break
    return hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()


def find_duplicates(
    items: list[Item], cfg: Settings, check_content: bool = True
) -> list[DuplicateGroup]:
    live = [it for it in items if it.triage is not Triage.SKIPPED]
    by_name = {it.name: it for it in live}
    same: list[tuple[Item, Item]] = []  # pasangan yang terbukti identik
    maybe: dict[Path, list[tuple[Item, str]]] = defaultdict(list)  # asli -> (salinan, catatan)

    # 1. Pola nama
    for dup in live:
        base = name_base(dup, cfg)
        orig = by_name.get(base) if base else None
        if orig is None or orig.kind is not dup.kind:
            continue
        if not check_content:
            maybe[orig.path].append((dup, "isi tidak diperiksa (--no-hash)"))
        elif dup.kind is ItemKind.FILE and orig.size_bytes != dup.size_bytes:
            maybe[orig.path].append((dup, "nama mirip, ukuran berbeda"))
        elif _identical(orig, dup, cfg):
            same.append((orig, dup))
        else:
            maybe[orig.path].append((dup, "nama mirip, isi berbeda"))

    # 2. Isi identik walau nama berbeda: saring per ukuran, lalu 64 KB awal, lalu hash penuh
    if check_content:
        by_size = defaultdict(list)
        for it in live:
            if it.kind is ItemKind.FILE and it.size_bytes > 0:
                by_size[it.size_bytes].append(it)
        for group in by_size.values():
            for quick in _bucket(group, lambda p: file_hash(p, cfg.quick_hash_bytes)):
                for full in _bucket(quick, file_hash):
                    same += [(full[0], other) for other in full[1:]]

    return _confirmed_groups(same, cfg) + [
        DuplicateGroup(
            original=orig,
            duplicates=[d.path for d, _ in dups],
            confirmed=False,
            note="; ".join(sorted({n for _, n in dups})),
        )
        for orig, dups in maybe.items()
    ]


def _identical(a: Item, b: Item, cfg: Settings) -> bool:
    if a.kind is ItemKind.FILE:
        return file_hash(a.path) == file_hash(b.path)
    return folder_signature(a.path, cfg) == folder_signature(b.path, cfg)


def _bucket(items: list[Item], key) -> list[list[Item]]:
    """Kelompokkan item berdasarkan key(path); hanya kelompok berisi 2 atau lebih."""
    if len(items) < 2:
        return []
    buckets = defaultdict(list)
    for it in items:
        buckets[key(it.path)].append(it)
    return [b for b in buckets.values() if len(b) > 1]


def _confirmed_groups(pairs: list[tuple[Item, Item]], cfg: Settings) -> list[DuplicateGroup]:
    """Gabungkan pasangan identik jadi grup (A=B dan B=C -> satu grup A, B, C)."""
    groups: list[dict[Path, Item]] = []
    for a, b in pairs:
        hits = [g for g in groups if a.path in g or b.path in g]
        merged = {a.path: a, b.path: b}
        for g in hits:
            merged |= g
            groups.remove(g)
        groups.append(merged)

    result = []
    for g in groups:
        # Asli: nama yang tidak berpola salinan, lalu yang tertua, lalu nama terpendek
        orig = min(
            g.values(), key=lambda it: (name_base(it, cfg) is not None, it.mtime, len(it.name))
        )
        result.append(
            DuplicateGroup(
                original=orig.path,
                duplicates=sorted(p for p in g if p != orig.path),
                confirmed=True,
                note="isi file identik"
                if orig.kind is ItemKind.FILE
                else "daftar isi folder identik",
                size_bytes=orig.size_bytes,
            )
        )
    return result