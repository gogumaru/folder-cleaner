"""Tes M3 tanpa Ollama: clustering dengan vektor buatan, dan loop agen dengan model palsu
yang memanggil tool sesuai skrip."""

from pathlib import Path

import numpy as np

from sorter.agent import Workspace, build_categories, run_agent
from sorter.cluster import agglomerate
from sorter.config import Settings
from sorter.describe import sample_with_copies
from sorter.scan import list_items
from sorter.schemas import Category, Cluster, Description, Descriptor
from sorter.store import Trace

CFG = Settings()


def test_agglomerate_groups_similar_vectors():
    x = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.1, 0.9, 0], [0, 0, 1]], dtype=float)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    groups = agglomerate(x, 3)
    assert sorted(map(sorted, groups)) == [[0, 1], [2, 3], [4]]


def _desc(name: str, doc_type: str) -> Descriptor:
    d = Description(summary=f"{doc_type} {name}", doc_type=doc_type, keywords=[], language="id")
    return Descriptor(path=Path(name), name=name, source="text", model="x", seconds=0,
                      description=d)  # fmt: skip


class ScriptedModel:
    """Mengembalikan tool call sesuai urutan skrip, dan mencatat pesan yang diterimanya."""

    name = "skrip"

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def chat(self, messages, tools, num_ctx):
        self.seen.append(list(messages))
        name, args = self.script.pop(0)
        return {"role": "assistant", "content": "",
                "tool_calls": [{"function": {"name": name, "arguments": args}}]}  # fmt: skip


def test_agent_loop_gets_feedback_and_finishes(tmp_path):
    descs = [_desc("a.pdf", "struk"), _desc("b.pdf", "struk"), _desc("c.jpg", "meme")]
    clusters = [Cluster(id=0, members=["a.pdf"]), Cluster(id=1, members=["b.pdf"]),
                Cluster(id=2, members=["c.jpg"])]  # fmt: skip
    keuangan = {"name": "Keuangan", "description": "Struk belanja dan bukti bayar harian",
                "tier": "important"}  # fmt: skip
    hiburan = {"name": "Hiburan", "description": "Meme dan gambar lucu untuk hiburan",
               "tier": "temporary", "cluster_ids": [2]}  # fmt: skip
    model = ScriptedModel([
        ("inspect_cluster", {"cluster_id": 0}),
        ("propose", {"categories": [keuangan | {"cluster_ids": [0, 9]}, hiburan]}),  # salah
        ("propose", {"categories": [keuangan | {"cluster_ids": [0, 1]}, hiburan]}),  # benar
    ])  # fmt: skip
    cfg = Settings(min_categories=2)
    ws = Workspace(clusters, descs, cfg)

    steps = run_agent(ws, model, cfg, Trace(tmp_path / "t.jsonl"))

    assert steps == 3
    assert [c.name for c in ws.final] == ["Keuangan", "Hiburan"]
    assert sorted(ws.final[0].items) == ["a.pdf", "b.pdf"]
    # propose yang salah dikembalikan ke model sebagai kesalahan, lalu model memperbaikinya
    last_tool_msg = model.seen[2][-1]
    assert last_tool_msg["role"] == "tool" and "tidak ada" in last_tool_msg["content"]


def test_repeated_failures_are_not_rerun_and_agent_is_reminded(tmp_path):
    descs = [_desc("a.pdf", "struk"), _desc("b.pdf", "struk")]
    ws = Workspace([Cluster(id=0, members=["a.pdf"]), Cluster(id=1, members=["b.pdf"])], descs,
                   Settings(min_categories=1))  # fmt: skip
    bad = ("split", {"cluster_id": 0, "items": ["a.pdf"]})  # semua isi cluster: gagal
    model = ScriptedModel([bad, bad, bad, ("propose", {"categories": [
        {"name": "Keuangan", "description": "Struk belanja dan bukti bayar harian",
         "tier": "important", "cluster_ids": [0, 1]}]})])  # fmt: skip
    events = []

    run_agent(ws, model, Settings(min_categories=1), Trace(tmp_path / "t.jsonl"),
              lambda kind, data: events.append(kind))  # fmt: skip

    assert "tidak perlu split" in model.seen[1][-1]["content"]
    assert "jangan diulang" in model.seen[2][-1]["content"]
    assert "stuck" in events and "Cluster saat ini" in model.seen[3][-1]["content"]
    assert ws.final is not None


def test_triage_categories_and_copies_are_added(sample):
    items = list_items(sample, CFG)
    _, copies = sample_with_copies(items, 400, CFG)
    agent_cats = build_categories([], items, copies)
    names = {c.name: c for c in agent_cats}
    assert "Visual Studio Code-2.app" in names["Aplikasi & Installer"].items
    assert names["Project"].tier == "important"
    assert "PPE-0" in names["Dataset"].items

    invoice = Category(name="Tagihan", description="x" * 20, tier="important",
                       items=["Invoice_Maret.pdf"])  # fmt: skip
    cats = build_categories([invoice], items, copies)
    assert "Invoice_Maret-2.pdf" in cats[0].items  # salinan ikut kategori file aslinya


def test_tool_errors_tell_the_agent_what_is_valid():
    descs = [_desc("a.pdf", "struk"), _desc("b.pdf", "struk"), _desc("c.jpg", "meme"),
             _desc("d.txt", "catatan"), _desc("e.py", "kode")]  # fmt: skip
    clusters = [Cluster(id=0, members=["a.pdf", "b.pdf"]),
                Cluster(id=2, members=["c.jpg", "d.txt", "e.py"])]  # fmt: skip
    ws = Workspace(clusters, descs, Settings(min_categories=2))

    stale = ws.run("split", {"cluster_id": 1, "items": ["a.pdf"]})
    assert "tidak ada" in stale["error"] and set(stale["cluster_aktif"]) == {"0", "2"}
    wrong = ws.run("split", {"cluster_id": 2, "items": ["kode"]})  # jenis, bukan nama file
    assert wrong["item_di_cluster"] == ["c.jpg", "d.txt", "e.py"]


def test_propose_warns_once_then_accepts():
    descs = [_desc("a", "struk"), _desc("b", "struk"), _desc("c", "struk"),
             _desc("d", "meme"), _desc("e", "catatan"), _desc("f", "kode")]  # fmt: skip
    ws = Workspace([Cluster(id=0, members=["a", "b"]), Cluster(id=1, members=["c"]),
                    Cluster(id=2, members=["d", "e", "f"])], descs,
                   Settings(min_categories=2))  # fmt: skip
    cat = {"description": "Struk belanja dan bukti bayar harian", "tier": "important"}
    cats = [cat | {"name": "A", "cluster_ids": [0]}, cat | {"name": "B", "cluster_ids": [1]},
            cat | {"name": "C", "cluster_ids": [2]}]  # fmt: skip

    first = ws.run("propose", {"categories": cats})
    assert first["diterima"] is False
    assert any("'B' hanya 1 item" in e for e in first["kesalahan"])
    assert any("cluster 2 campuran" in e for e in first["kesalahan"])
    second = ws.run("propose", {"categories": cats})  # agen tetap pada keputusannya
    assert second["diterima"] is True and len(second["peringatan"]) == 2


def test_agent_can_resend_the_same_proposal_after_a_warning(tmp_path):
    descs = [_desc("a", "struk"), _desc("b", "struk"), _desc("c", "meme")]
    cfg = Settings(min_categories=1)
    ws = Workspace([Cluster(id=0, members=["a", "b"]), Cluster(id=1, members=["c"])], descs, cfg)
    cat = {"description": "Struk belanja dan bukti bayar harian", "tier": "important"}
    same = ("propose", {"categories": [cat | {"name": "A", "cluster_ids": [0]},
                                       cat | {"name": "B", "cluster_ids": [1]}]})  # fmt: skip
    steps = run_agent(ws, ScriptedModel([same, same]), cfg, Trace(tmp_path / "t.jsonl"))
    assert steps == 2 and ws.final is not None