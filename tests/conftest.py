"""Dipakai bersama oleh semua tes: folder contoh dan snapshot isi folder."""

import os
from pathlib import Path

import pytest

from sorter.sample import make_sample_folder


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    return make_sample_folder(tmp_path / "sample")


def snapshot(root: Path) -> dict:
    """Semua path di bawah root beserta ukuran, mtime, dan mode."""
    snap = {".": os.lstat(root).st_mtime_ns}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            st = os.lstat(p)
            snap[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns, st.st_mode)
    return snap