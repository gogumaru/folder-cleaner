"""Tes M2 tanpa Ollama: model diganti model palsu yang mencatat apa yang dikirim kepadanya."""

import os
import zipfile
from pathlib import Path
import pytest
from conftest import snapshot

from sorter.config import Settings
from sorter.describe import NOT_READ, describe_items, pick_sample, sample_with_copies
from sorter.extract import extract
from sorter.llm import ModelUnavailable, Ollama
from sorter.scan import list_items
from sorter.schemas import Description, Descriptor, Item
from sorter.store import Cache, Trace

CFG = Settings()


class FakeModel:
    name = "palsu"

    def __init__(self):
        self.calls = []

    def describe(self, prompt, image=None):
        self.calls.append((prompt, image))
        return Description(summary="contoh", doc_type="contoh", keywords=[], language="id")


def test_describe_is_read_only_and_uses_cache(sample, tmp_path):
    before = snapshot(sample)
    items = pick_sample(list_items(sample, CFG), 400)
    cache = Cache(tmp_path / "cache.sqlite")
    model = FakeModel()

    first = describe_items(items, CFG, model, cache, Trace(tmp_path / "t1.jsonl"))
    second = describe_items(items, CFG, model, cache, Trace(tmp_path / "t2.jsonl"))

    assert snapshot(sample) == before  # tidak ada yang berubah di folder
    read = [d for d in first if d.source != "name_only"]
    assert len(model.calls) == len(read)  # run kedua tidak memanggil model sama sekali
    assert not any(d.from_cache for d in first)
    assert all(d.from_cache for d in second if d.source != "name_only")
    assert not any(d.from_cache for d in second if d.source == "name_only")  # tidak di-cache


def test_model_gets_the_right_snippet(sample):
    items = {it.name: it for it in list_items(sample, CFG)}
    got = {name: extract(items[name], CFG) for name in items if items[name].triage == "needs_model"}

    assert got["Invoice_Maret.pdf"].source == "text"
    assert "INV-2026-03" in got["Invoice_Maret.pdf"].text
    assert got["scan0001.pdf"].source == "image"  # PDF hasil scan tanpa teks -> dirender
    assert got["IMG_4821.jpg"].image[:2] == b"\xff\xd8"  # JPEG
    assert "Bakti Sosial" in got["laporan.docx"].text
    assert "tugas1.pdf" in got["Tugas Kuliah"].text  # folder -> daftar nama saja
    assert got["backup.dat"].source == "name_only"


def test_only_local_models_are_allowed():
    with pytest.raises(ModelUnavailable):
        Ollama("http://192.168.1.5:11434", "qwen3-vl:8b", 10)
    with pytest.raises(ModelUnavailable):
        Ollama("http://127.0.0.1:11434", "qwen3-vl:235b-cloud", 10)
    Ollama(Settings().ollama_host, "qwen3-vl:8b", 10)  # default: lokal, tidak error

def test_model_never_guesses_from_filename_alone(sample, tmp_path):
    items = {it.name: it for it in list_items(sample, CFG)}
    chosen = [items["IMG_4821.jpg"], items["backup.dat"]]
    model = FakeModel()

    results = describe_items(
        chosen, CFG, model, Cache(tmp_path / "c.sqlite"), Trace(tmp_path / "t")
    )

    [(prompt, image)] = model.calls  # backup.dat (name_only) tidak dikirim ke model
    assert image is not None and "IMG_4821" not in prompt  # gambar dinilai tanpa nama file
    assert results[1].description.doc_type == "tidak dibaca"

def test_identical_copies_are_not_sent_to_model(sample, tmp_path):
    chosen, copies = sample_with_copies(list_items(sample, CFG), 400, CFG)
    model = FakeModel()

    results = describe_items(
        chosen, CFG, model, Cache(tmp_path / "c.sqlite"), Trace(tmp_path / "t"), copies=copies
    )

    by_name = {d.name: d for d in results}
    pairs = [
        ("Invoice_Maret-2.pdf", "Invoice_Maret.pdf"),
        ("meme (1).jpg", "meme.jpg"),
        ("salinan-scan.pdf", "scan0001.pdf"),
    ]
    for copy, original in pairs:
        assert by_name[copy].duplicate_of.name == original
        assert by_name[copy].description == by_name[original].description
    sent = [d for d in results if d.source != "name_only" and not d.duplicate_of]
    assert len(model.calls) == len(sent)  # salinan tidak pernah dikirim ke model



def _zip(path, files: dict) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def _item(path: Path) -> Item:
    return next(it for it in list_items(path.parent, CFG) if it.name == path.name)


def test_office_files_are_read_as_text(tmp_path):
    slide = '<p:sld><a:t>Strategi Bisnis</a:t><a:t>Minggu 3</a:t></p:sld>'
    pptx = _zip(tmp_path / "kuliah.pptx", {"ppt/slides/slide2.xml": "<a:t>Analisis SWOT</a:t>",
                                            "ppt/slides/slide1.xml": slide})  # fmt: skip
    xlsx = _zip(tmp_path / "data.xlsx", {
        "xl/workbook.xml": '<sheets><sheet name="Nasabah" sheetId="1"/></sheets>',
        "xl/sharedStrings.xml": "<sst><si><t>Nama</t></si><si><t>Saldo</t></si></sst>",
    })  # fmt: skip
    for path in (pptx, xlsx):
        os.utime(path, (0, 0))  # supaya tidak dilewati sebagai "terlalu baru"

    p = extract(_item(pptx), CFG)
    x = extract(_item(xlsx), CFG)

    assert p.source == "text" and p.text.startswith("[slide 1] Strategi Bisnis Minggu 3")
    assert "[slide 2] Analisis SWOT" in p.text
    assert x.source == "text" and "Nasabah" in x.text and "Nama | Saldo" in x.text


def test_unread_items_are_not_served_from_cache(sample, tmp_path):
    items = [it for it in list_items(sample, CFG) if it.name == "backup.dat"]
    cache = Cache(tmp_path / "cache.sqlite")
    old = Descriptor(path=items[0].path, name="backup.dat", source="name_only", model="palsu",
                     seconds=0, description=NOT_READ)  # fmt: skip
    cache.put(items[0], old)  # seperti cache dari versi lama, sebelum ada perbaikan

    result = describe_items(items, CFG, FakeModel(), cache, Trace(tmp_path / "t.jsonl"))

    assert not result[0].from_cache  # dibaca ulang, siapa tahu formatnya sekarang didukung