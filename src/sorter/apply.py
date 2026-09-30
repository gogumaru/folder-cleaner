"""Menjalankan plan.json di playground, dan membatalkannya (M4).

Playground adalah salinan folder asli: <target>/Downloads berisi salinan folder yang dipindai,
<target>/Documents awalnya kosong. Di Mac salinannya memakai APFS clone (cp -c), jadi instan dan
tidak memakan ruang. Folder asli tidak pernah disentuh.

Setiap langkah dicatat di journal SEBELUM dikerjakan, supaya undo tetap bisa jalan walau proses
berhenti di tengah. Tidak ada yang dihapus atau ditimpa.
"""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from sorter.sample import PROTECTED
from sorter.schemas import Plan

MARKER = ".sorter-playground"
ROOTS = ("Documents", "Downloads")


class ApplyError(Exception):
    pass


def make_playground(source: Path, target: Path) -> Path:
    """Buat playground dari `source`, atau pakai lagi playground lama dari source yang sama."""
    source, target = source.resolve(), target.expanduser().resolve()
    if not source.is_dir():
        raise ApplyError(f"Folder asli tidak ditemukan: {source}")
    home = Path.home().resolve()
    for name in PROTECTED:
        if target == home / name or target.is_relative_to(home / name):
            raise ApplyError(f"Playground tidak boleh di dalam ~/{name}.")
    if target == home or target.is_relative_to(source) or source.is_relative_to(target):
        raise ApplyError(f"{target} bertumpuk dengan folder asli {source}. Pilih lokasi lain.")
    marker = target / MARKER
    if target.exists():
        if not marker.is_file():
            raise ApplyError(f"{target} sudah ada dan bukan playground. Pilih lokasi lain.")
        if json.loads(marker.read_text())["source"] != str(source):
            raise ApplyError(f"{target} adalah playground untuk folder lain.")
        return target
    target.mkdir(parents=True)
    _clone(source, target / "Downloads")
    (target / "Documents").mkdir()
    marker.write_text(json.dumps({"source": str(source), "created": datetime.now().isoformat()}))
    return target


def _clone(src: Path, dst: Path) -> None:
    if sys.platform == "darwin":
        try:  # -c: APFS clone. Gagal bila bukan APFS, lalu pakai salinan biasa.
            subprocess.run(["cp", "-cR", str(src), str(dst)], check=True, capture_output=True)
            return
        except subprocess.CalledProcessError:
            shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, symlinks=True)


def apply_plan(
    plan: Plan, playground: Path, journal: Path, on_move: Callable[[str, str], None] | None = None
) -> dict[str, int]:
    """Pindahkan item sesuai rencana di dalam playground. Mengembalikan jumlah per hasil."""
    root = playground.resolve()
    if not (root / MARKER).is_file():
        raise ApplyError(f"{root} bukan playground.")
    if journal.exists():
        raise ApplyError("Rencana ini sudah dijalankan. Jalankan undo dulu bila mau mengulang.")
    moves = [p for p in plan.items if p.action == "move"]
    for p in moves:
        _check_name(p.name)
        _check_dest(p.dest)
    stats = {"dipindah": 0, "tidak_ditemukan": 0}
    folders: dict[str, Path] = {}  # dest di rencana -> folder yang benar-benar dipakai
    with journal.open("w") as log:
        _log(log, {"op": "start", "playground": str(root), "run_id": plan.run_id})
        for p in moves:
            src = root / "Downloads" / p.name
            if not os.path.lexists(src):
                stats["tidak_ditemukan"] += 1
                continue
            if p.dest not in folders:
                folders[p.dest] = _new_folder(root, p.dest, log)
            dst = _free_name(folders[p.dest] / p.name)
            _log(log, {"op": "move", "src": _rel(src, root), "dst": _rel(dst, root)})
            os.rename(src, dst)
            stats["dipindah"] += 1
            if on_move:
                on_move(p.name, _rel(dst.parent, root))
    return stats


def undo(journal: Path) -> dict[str, int]:
    """Kembalikan semua pemindahan di journal, urut dari yang terakhir."""
    if not journal.exists():
        raise ApplyError("Tidak ada journal: rencana ini belum dijalankan atau sudah di-undo.")
    entries = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    root = Path(entries[0]["playground"])
    if not (root / MARKER).is_file():
        raise ApplyError(f"{root} bukan playground.")
    stats = {"dikembalikan": 0, "bentrok": 0}
    for e in reversed(entries[1:]):
        if e["op"] == "move":
            src, dst = root / e["src"], root / e["dst"]
            if os.path.lexists(dst) and not os.path.lexists(src):
                os.rename(dst, src)
                stats["dikembalikan"] += 1
            else:  # sudah dipindah orang lain, atau tempat asalnya terisi: jangan menimpa
                stats["bentrok"] += 1
        elif e["op"] == "mkdir":
            try:
                (root / e["path"]).rmdir()  # hanya bila kosong
            except OSError:
                pass
    journal.rename(journal.with_name(f"journal-undone-{datetime.now():%Y%m%d-%H%M%S}.jsonl"))
    return stats


def _check_name(name: str) -> None:
    if "/" in name or name in ("", ".", ".."):
        raise ApplyError(f"Nama item tidak valid di rencana: {name!r}")


def _check_dest(dest: str | None) -> None:
    parts = Path(dest or "").parts
    if len(parts) != 2 or parts[0] not in ROOTS or parts[1] in (".", ".."):
        raise ApplyError(
            f"Tujuan tidak valid di rencana: {dest!r}. Harus Documents/<Kategori> atau "
            "Downloads/<Kategori>, satu level saja."
        )


def _new_folder(root: Path, dest: str, log) -> Path:
    """Folder kategori baru. Bila namanya sudah dipakai item lain, beri akhiran, jangan campur."""
    folder = _free_name(root / dest)
    folder.mkdir(parents=True)
    _log(log, {"op": "mkdir", "path": _rel(folder, root)})
    return folder


def _free_name(path: Path) -> Path:
    """Nama yang belum dipakai: "a.pdf", lalu "a (2).pdf", "a (3).pdf", dst. Tidak menimpa."""
    if not os.path.lexists(path):
        return path
    stem, suffix = (path.stem, path.suffix) if path.is_file() else (path.name, "")
    n = 2
    while os.path.lexists(path.with_name(f"{stem} ({n}){suffix}")):
        n += 1
    return path.with_name(f"{stem} ({n}){suffix}")


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _log(log, entry: dict) -> None:
    log.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.flush()
    os.fsync(log.fileno())  # benar-benar tertulis ke disk sebelum file dipindah