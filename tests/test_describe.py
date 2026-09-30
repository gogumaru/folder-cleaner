"""Tes M2 tanpa Ollama: model diganti model palsu yang mencatat apa yang dikirim kepadanya."""

import pytest
from conftest import snapshot

from sorter.config import Settings
from sorter.describe import describe_items, pick_sample, sample_with_copies
from sorter.extract import extract
from sorter.llm import ModelUnavailable, Ollama
from sorter.scan import list_items
from sorter.schemas import Description
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
    assert all(d.from_cache for d in second)


def test_model_gets_the_right_snippet(sample):
    items = {it.name: it for it in list_items(sample, CFG)}
    got = {name: extract(items[name], CFG) for name in items if items[name].triage == "needs_model"}

    assert got["Invoice_Maret.pdf"].source == "text"
    assert "INV-2026-03" in got["Invoice_Maret.pdf"].text
    assert got["scan0001.pdf"].source == "image"  # PDF hasil scan tanpa teks -> dirender
    assert got["IMG_4821.jpg"].image[:2] == b"\xff\xd8"  # JPEG
    assert "Bakti Sosial" in got["laporan.docx"].text
    assert "tugas1.pdf" in got["Tugas Kuliah"].text  # folder -> daftar nama saja
    assert got["musik.mp3"].source == "name_only"


def test_only_local_models_are_allowed():
    with pytest.raises(ModelUnavailable):
        Ollama("http://192.168.1.5:11434", "qwen3-vl:8b", 10)
    with pytest.raises(ModelUnavailable):
        Ollama("http://127.0.0.1:11434", "qwen3-vl:235b-cloud", 10)
    Ollama(Settings().ollama_host, "qwen3-vl:8b", 10)  # default: lokal, tidak error

def test_model_never_guesses_from_filename_alone(sample, tmp_path):
    items = {it.name: it for it in list_items(sample, CFG)}
    chosen = [items["IMG_4821.jpg"], items["musik.mp3"]]
    model = FakeModel()

    results = describe_items(
        chosen, CFG, model, Cache(tmp_path / "c.sqlite"), Trace(tmp_path / "t")
    )

    [(prompt, image)] = model.calls  # musik.mp3 (name_only) tidak dikirim ke model
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