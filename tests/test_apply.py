"""Tes M4: rencana pemindahan, apply di playground, dan undo. Tanpa Ollama."""

import json
import stat
from pathlib import Path

import pytest

from conftest import snapshot
from sorter.agent import build_categories
from sorter.apply import ApplyError, apply_plan, make_playground, undo
from sorter.config import Settings
from sorter.describe import sample_with_copies
from sorter.plan import build_plan
from sorter.scan import list_items
from sorter.schemas import Category, Description, Descriptor, Plan, PlanItem, Taxonomy

CFG = Settings()


def _plan(sample: Path) -> Plan:
    """Rencana seperti hasil discover: 2 kategori dari agen + kategori tetap dari triage."""
    items = list_items(sample, CFG)
    _, copies = sample_with_copies(items, 400, CFG)
    agent = [
        Category(name="Keuangan", description="x" * 20, tier="important",
                 items=["Invoice_Maret.pdf", "kwitansi.pdf"]),
        Category(name="Hiburan", description="x" * 20, tier="temporary", items=["meme.jpg"]),
    ]  # fmt: skip
    tax = Taxonomy(run_id="r1", folder=sample, model="x", unread=["musik.mp3"], agent_steps=1,
                   agent_finished=True, categories=build_categories(agent, items, copies))  # fmt: skip
    descs = [
        Descriptor(path=sample / n, name=n, source="text", model="x", seconds=0,
                   description=Description(summary="ringkasan", doc_type=t, keywords=[],
                                           language="id"))
        for n, t in [("Invoice_Maret.pdf", "invoice"), ("kwitansi.pdf", "kwitansi"),
                     ("meme.jpg", "meme"), ("todo.txt", "catatan")]
    ]  # fmt: skip
    return build_plan(tax, items, descs, copies)


def test_plan_says_where_and_why(sample):
    rows = {p.name: p for p in _plan(sample).items}

    assert rows["Invoice_Maret.pdf"].dest == "Documents/Keuangan"
    assert rows["Invoice_Maret.pdf"].reason == "invoice: ringkasan"
    copy = rows["Invoice_Maret-2.pdf"]  # salinan ikut aslinya, ditandai untuk dihapus user
    assert copy.dest == "Documents/Keuangan" and copy.duplicate_of == "Invoice_Maret.pdf"
    assert rows["meme.jpg"].dest == "Downloads/Hiburan"
    assert rows["Docker.dmg"].dest == "Downloads/Aplikasi & Installer"
    assert rows["Docker.dmg"].source == "aturan"
    assert rows["musik.mp3"].action == "stay" and "tidak terbaca" in rows["musik.mp3"].reason
    assert rows["todo.txt"].action == "stay"  # dibaca, tapi tidak masuk kategori mana pun
    assert rows["scan0002.pdf"].action == "stay" and "di luar sampel" in rows["scan0002.pdf"].reason
    assert rows[".DS_Store"].action == "stay"


def _files(root: Path) -> dict:
    """Isi folder tanpa waktu ubah folder (yang pasti berubah saat isinya dipindah)."""
    return {k: v for k, v in snapshot(root).items() if k != "." and not stat.S_ISDIR(v[2])}


def test_apply_then_undo_restores_everything(sample, tmp_path):
    original = snapshot(sample)
    plan = _plan(sample)
    playground = make_playground(sample, tmp_path / "pg")
    fresh = _files(playground / "Downloads")
    journal = tmp_path / "journal.jsonl"

    stats = apply_plan(plan, playground, journal)

    moved = sum(p.action == "move" for p in plan.items)
    assert stats == {"dipindah": moved, "tidak_ditemukan": 0}
    assert (playground / "Documents/Keuangan/Invoice_Maret-2.pdf").is_file()
    assert (playground / "Downloads/Aplikasi & Installer/Visual Studio Code.app").is_dir()
    assert not (playground / "Downloads/Invoice_Maret.pdf").exists()
    assert (playground / "Downloads/musik.mp3").is_file()  # "tetap" tidak disentuh
    assert snapshot(sample) == original  # folder asli sama sekali tidak berubah

    assert undo(journal) == {"dikembalikan": moved, "bentrok": 0}
    assert _files(playground / "Downloads") == fresh
    assert list((playground / "Documents").iterdir()) == []  # folder kategori kosong dihapus
    assert snapshot(sample) == original


def test_never_merges_or_overwrites(sample, tmp_path):
    playground = make_playground(sample, tmp_path / "pg")
    plan = Plan(run_id="r", folder=sample, items=[  # kategori bernama sama dengan folder user
        PlanItem(name="catatan.txt", action="move", dest="Downloads/Tugas Kuliah", reason="x"),
    ])  # fmt: skip
    before = _files(playground / "Downloads/Tugas Kuliah")

    apply_plan(plan, playground, tmp_path / "j.jsonl")

    assert (playground / "Downloads/Tugas Kuliah (2)/catatan.txt").is_file()
    assert _files(playground / "Downloads/Tugas Kuliah") == before


def test_refuses_unsafe_targets_and_plans(sample, tmp_path):
    with pytest.raises(ApplyError, match="Downloads"):
        make_playground(sample, Path.home() / "Downloads" / "pg")
    with pytest.raises(ApplyError, match="bertumpuk"):
        make_playground(sample, sample / "pg")
    (tmp_path / "punyaku").mkdir()
    with pytest.raises(ApplyError, match="bukan playground"):
        make_playground(sample, tmp_path / "punyaku")

    playground = make_playground(sample, tmp_path / "pg")
    for dest in ["../../etc", "Desktop/X", "Documents/A/B", "Documents/.."]:
        bad = Plan(run_id="r", folder=sample, items=[
            PlanItem(name="catatan.txt", action="move", dest=dest, reason="x")])  # fmt: skip
        with pytest.raises(ApplyError, match="Tujuan tidak valid"):
            apply_plan(bad, playground, tmp_path / "j.jsonl")

    ok = Plan(run_id="r", folder=sample, items=[
        PlanItem(name="catatan.txt", action="move", dest="Downloads/Catatan", reason="x")])  # fmt: skip
    apply_plan(ok, playground, tmp_path / "j.jsonl")
    with pytest.raises(ApplyError, match="sudah dijalankan"):
        apply_plan(ok, playground, tmp_path / "j.jsonl")


def test_undo_does_not_overwrite_a_refilled_spot(sample, tmp_path):
    playground = make_playground(sample, tmp_path / "pg")
    journal = tmp_path / "j.jsonl"
    plan = Plan(run_id="r", folder=sample, items=[
        PlanItem(name="catatan.txt", action="move", dest="Downloads/Catatan", reason="x")])  # fmt: skip
    apply_plan(plan, playground, journal)
    (playground / "Downloads/catatan.txt").write_text("file baru dengan nama sama")

    assert undo(journal) == {"dikembalikan": 0, "bentrok": 1}
    assert (playground / "Downloads/catatan.txt").read_text() == "file baru dengan nama sama"
    assert (playground / "Downloads/Catatan/catatan.txt").is_file()
    assert json.loads((playground / ".sorter-playground").read_text())["source"] == str(sample)