"""Tes yang menjaga dua janji M1:
1. scan tidak mengubah apa pun di folder yang dipindai;
2. duplikat yang dilaporkan "identik" memang identik.
"""

import json
import os

from conftest import snapshot
from typer.testing import CliRunner

from sorter import cli
from sorter.config import Settings
from sorter.duplicates import find_duplicates

from sorter.scan import list_items
from sorter.schemas import Triage

CFG = Settings()




# ---------- Janji 1: baca-saja ----------


def test_scan_does_not_change_scanned_folder(sample, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "SETTINGS", Settings(runs_dir=tmp_path / "runs"))
    before = snapshot(sample)

    result = CliRunner().invoke(cli.app, ["scan", str(sample), "--details"])

    assert result.exit_code == 0, result.output
    assert snapshot(sample) == before
    [run] = (tmp_path / "runs").iterdir()
    assert len(json.loads((run / "inventory.json").read_text())["items"]) == len(os.listdir(sample))


def test_scan_refuses_output_inside_scanned_folder(sample, monkeypatch):
    monkeypatch.chdir(sample)  # runs_dir "runs" relatif -> akan jatuh di dalam folder contoh
    before = snapshot(sample)

    result = CliRunner().invoke(cli.app, ["scan", "."])

    assert result.exit_code == 2
    assert snapshot(sample) == before


# ---------- Janji 2: duplikat ----------


def groups(sample, check_content=True):
    items = list_items(sample, CFG)
    return {g.original.name: g for g in find_duplicates(items, CFG, check_content)}


def test_identical_duplicates_are_confirmed(sample):
    g = groups(sample)
    assert g["Invoice_Maret.pdf"].confirmed  # file, pola "-2"
    assert g["meme.jpg"].confirmed  # file, pola "(1)"
    assert g["Innowork-24"].confirmed  # folder, pola "-2"
    apps = g["Visual Studio Code.app"]
    assert apps.confirmed and [p.name for p in apps.duplicates] == [
        "Visual Studio Code-2.app",
        "Visual Studio Code-3.app",
    ]
    # isi sama walau namanya tidak mirip
    scan = g.get("scan0001.pdf") or g["salinan-scan.pdf"]
    assert scan.confirmed


def test_similar_name_but_different_content_is_not_confirmed(sample):
    g = groups(sample)
    assert not g["laporan.docx"].confirmed
    assert not g["maskYOLOv5"].confirmed


def test_no_hash_never_confirms(sample):
    g = groups(sample, check_content=False)
    assert g and not any(grp.confirmed for grp in g.values())


# ---------- Triage ----------


def test_triage_labels(sample):
    got = {it.name: it.triage for it in list_items(sample, CFG)}
    assert got["Visual Studio Code.app"] is Triage.APP
    assert got["Docker.dmg"] is Triage.INSTALLER
    assert got["logs.tar.gz"] is Triage.ARCHIVE
    assert got["yolov8-silva-main"] is Triage.PROJECT  # project menang atas dataset
    assert got["Face Detection"] is Triage.PROJECT  # penanda di folder pembungkus
    assert got["PPE-0"] is Triage.DATASET
    assert got["Tugas Kuliah"] is Triage.NEEDS_MODEL
    for skipped in ("film.mp4.crdownload", ".DS_Store", ".kontrak.pdf.icloud", "baru-diunduh.pdf"):
        assert got[skipped] is Triage.SKIPPED, skipped