"""Scan + triage: daftar item level teratas beserta labelnya, tanpa model.

BACA-SAJA: hanya membaca daftar isi dan metadata. Tidak ada yang ditulis, dipindah,
diganti nama, atau dihapus di folder yang dipindai. Isi folder tidak dibongkar; yang dibaca
hanya daftar nama di level pertama (dan satu level lagi bila folder hanya pembungkus unzip).
"""

import os
from datetime import datetime
from pathlib import Path

from sorter.config import Settings
from sorter.schemas import Item, ItemKind, Triage


class UnsafeOutputError(Exception):
    pass


def assert_outside(output_dir: Path, folder: Path) -> None:
    """Tolak bila output akan ditulis di dalam folder yang dipindai."""
    out, src = output_dir.expanduser().resolve(), folder.expanduser().resolve()
    if out == src or out.is_relative_to(src):
        raise UnsafeOutputError(
            f"Folder output {out} ada di dalam {src}. Scan tidak boleh menulis ke folder yang "
            "dipindai. Jalankan dari folder project atau atur SORTER_RUNS_DIR."
        )


def list_items(folder: Path, cfg: Settings, now: datetime | None = None) -> list[Item]:
    now = now or datetime.now()
    with os.scandir(folder) as it:
        entries = sorted(it, key=lambda e: e.name.lower())
    items = []
    for e in entries:
        try:
            st = e.stat(follow_symlinks=False)
        except OSError:
            continue
        kind = _kind(e, cfg)
        suffix = _suffix(e.name, kind, cfg)
        mtime = datetime.fromtimestamp(st.st_mtime)
        label, reason = triage(Path(e.path), e.name, kind, suffix, mtime, now, cfg)
        items.append(
            Item(
                path=Path(e.path),
                name=e.name,
                kind=kind,
                suffix=suffix,
                size_bytes=st.st_size if kind is ItemKind.FILE else 0,
                mtime=mtime,
                triage=label,
                reason=reason,
            )
        )
    return items


def triage(
    path: Path, name: str, kind: ItemKind, suffix: str, mtime: datetime, now: datetime,
    cfg: Settings,
) -> tuple[Triage, str]:  # fmt: skip
    lower = name.lower()
    if lower.endswith(".icloud"):
        return Triage.SKIPPED, "placeholder iCloud, file belum ada di Mac"
    if name in cfg.ignored_names or name.startswith("."):
        return Triage.SKIPPED, "item tersembunyi atau file sistem"
    if lower.endswith(cfg.partial_suffixes):
        return Triage.SKIPPED, "unduhan belum selesai"
    if kind is ItemKind.SYMLINK:
        return Triage.SKIPPED, "symlink tidak diikuti"
    if (now - mtime).total_seconds() < cfg.min_age_seconds:
        return Triage.SKIPPED, "terlalu baru, mungkin masih ditulis"

    if suffix == ".app":
        return Triage.APP, "bundle aplikasi"
    if suffix in cfg.installer_suffixes:
        return Triage.INSTALLER, f"installer {suffix}"
    if suffix in cfg.archive_suffixes:
        return Triage.ARCHIVE, f"arsip {suffix}"
    for suffixes, label, what in (
        (cfg.video_suffixes, Triage.VIDEO, "video"),
        (cfg.audio_suffixes, Triage.AUDIO, "audio"),
        (cfg.font_suffixes, Triage.FONT, "font"),
        (cfg.model_suffixes, Triage.MODEL, "model ML"),
    ):
        if kind is ItemKind.FILE and suffix in suffixes:
            return label, f"{what} {suffix}"

    if kind is ItemKind.FOLDER:
        for place, names in _places(path):
            where = "" if place == path else f" (di {place.name}/)"
            found = [m for m in cfg.project_markers if m.lower() in names]
            if found:
                return Triage.PROJECT, f"penanda project: {', '.join(found)}{where}"
            found = [m for m in cfg.dataset_markers if m.lower() in names]
            if found:
                return Triage.DATASET, f"penanda dataset: {', '.join(found)}{where}"
            for pattern in cfg.dataset_dir_patterns:
                if all(p in names for p in pattern):
                    return Triage.DATASET, f"pola dataset: {' + '.join(pattern)}{where}"
        return Triage.NEEDS_MODEL, "folder tanpa penanda project atau dataset"

    return Triage.NEEDS_MODEL, "perlu dibaca isinya"


def _kind(e: os.DirEntry, cfg: Settings) -> ItemKind:
    if e.is_symlink():
        return ItemKind.SYMLINK
    if e.is_dir(follow_symlinks=False):
        return ItemKind.BUNDLE if e.name.lower().endswith(cfg.package_suffixes) else ItemKind.FOLDER
    return ItemKind.FILE


def _suffix(name: str, kind: ItemKind, cfg: Settings) -> str:
    if kind is ItemKind.FOLDER:
        return ""
    lower = name.lower()
    for multi in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if lower.endswith(multi):
            return multi
    return Path(lower).suffix


def _names(folder: Path) -> tuple[set[str], list[Path]]:
    """Nama di level pertama (huruf kecil) dan daftar subfolder yang tidak tersembunyi."""
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return set(), []
    names = {e.name.lower() for e in entries}
    subdirs = [
        Path(e.path)
        for e in entries
        if e.is_dir(follow_symlinks=False) and not e.name.startswith(".")
    ]
    return names, subdirs


def _places(folder: Path) -> list[tuple[Path, set[str]]]:
    """Tempat mencari penanda: folder itu sendiri, dan isinya bila hanya berisi satu subfolder
    (pola hasil unzip seperti "nama/nama-main/...")."""
    names, subdirs = _names(folder)
    places = [(folder, names)]
    loose_files = len([n for n in names if not n.startswith(".")]) - len(subdirs)
    if len(subdirs) == 1 and loose_files <= 3:
        places.append((subdirs[0], _names(subdirs[0])[0]))
    return places