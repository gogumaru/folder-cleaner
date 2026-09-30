"""Tes perintah CLI dari ujung ke ujung dengan Ollama palsu, supaya salah ketik di cli.py
(misal fungsi atau import yang terlewat) ketahuan tanpa harus menjalankan model sungguhan."""

import json
import re

from typer.testing import CliRunner

from conftest import snapshot
from sorter import cli
from sorter.llm import ModelError
from sorter.schemas import Description


class FakeOllama:
    """Meniru Ollama: describe selalu menjawab sama, agen langsung propose 3 kategori."""

    def __init__(self, host, model, timeout):
        self.name = model

    def check(self, model=None):
        pass

    def describe(self, prompt, image=None):
        return Description(summary="contoh", doc_type="contoh", keywords=[], language="id")

    def embed(self, model, texts):
        return [[1.0, float(i % 3)] for i in range(len(texts))]

    def chat(self, messages, tools, num_ctx):
        ids = [int(i) for i in re.findall(r"Cluster (\d+)", messages[1]["content"])]
        cats = [{"name": f"Kategori {k}", "description": "Deskripsi yang cukup panjang",
                 "tier": "important", "cluster_ids": ids[k::3]} for k in range(3)]  # fmt: skip
        call = {"function": {"name": "propose", "arguments": {"categories": cats}}}
        return {"role": "assistant", "content": "", "tool_calls": [call]}

class FakeApple:
    """Meniru model Apple: selalu memilih kategori pertama."""

    name = "apple-palsu"

    def choose(self, instructions, prompt, image, choices):
        return {"category": choices[0], "confidence": "sedang", "reason": "contoh"}

def test_describe_and_discover_run_end_to_end(sample, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # runs/ dan cache dibuat di folder tes
    monkeypatch.setattr(cli, "Ollama", FakeOllama)
    before = snapshot(sample)
    runner = CliRunner()

    described = runner.invoke(cli.app, ["describe", str(sample)])
    assert described.exit_code == 0, described.output
    discovered = runner.invoke(cli.app, ["discover", str(sample), "--agent-model", "lain"])
    assert discovered.exit_code == 0, discovered.output

    taxonomy = json.loads(next(tmp_path.glob("runs/*/taxonomy.json")).read_text())
    names = [c["name"] for c in taxonomy["categories"]]
    assert names[:3] == ["Kategori 0", "Kategori 1", "Kategori 2"]
    assert "Aplikasi & Installer" in names  # kategori tetap dari triage

    run = next(tmp_path.glob("runs/*/plan.json")).parent
    playground = tmp_path / "pg"
    applied = runner.invoke(cli.app, ["apply", str(run), "--target", str(playground), "--yes"])
    assert applied.exit_code == 0, applied.output
    assert (playground / "Documents/Kategori 0").is_dir()
    undone = runner.invoke(cli.app, ["undo", str(run)])
    assert undone.exit_code == 0, undone.output
    assert list((playground / "Documents").iterdir()) == []

    monkeypatch.setattr(cli, "AppleModel", FakeApple)
    classified = runner.invoke(cli.app, ["classify", str(run)])
    assert classified.exit_code == 0, classified.output
    assert "Sama dengan agen" in classified.output
    one = runner.invoke(cli.app, ["classify", str(run), "--item", str(sample / "catatan.txt")])
    assert one.exit_code == 0 and "catatan.txt" in one.output
    assert snapshot(sample) == before


class SlowAgent(FakeOllama):
    def chat(self, messages, tools, num_ctx):
        raise ModelError("tidak menjawab dalam 600 detik")


def test_discover_reports_model_timeout_without_traceback(sample, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "Ollama", SlowAgent)
    result = CliRunner().invoke(cli.app, ["discover", str(sample)])
    assert result.exit_code == 2  # _fail: pesan merah, bukan traceback
    assert "tidak menjawab" in result.output and "Traceback" not in result.output