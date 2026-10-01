"""Tes M3 tanpa Ollama: clustering dengan vektor buatan, agen perancang kategori dengan model
palsu yang memanggil tool sesuai skrip, dan langkah pilah dengan pemilih palsu."""

from pathlib import Path

import numpy as np

from sorter.agent import NONE, Workspace, build_categories, run_agent
from sorter.assign import assign
from sorter.cluster import agglomerate, cluster_descriptors
from sorter.config import Settings
from sorter.describe import sample_with_copies
from sorter.llm import ModelError
from sorter.scan import list_items
from sorter.schemas import Category, Cluster, Description, Descriptor
from sorter.store import Trace

CFG = Settings()
GOOD = "Struk belanja, invoice, dan bukti bayar harian"


def test_agglomerate_groups_similar_vectors():
    x = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.1, 0.9, 0], [0, 0, 1]], dtype=float)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    groups = agglomerate(x, 3)
    assert sorted(map(sorted, groups)) == [[0, 1], [2, 3], [4]]


def _desc(name: str, doc_type: str, source: str = "text") -> Descriptor:
    d = Description(summary=f"{doc_type} {name}", doc_type=doc_type, keywords=[], language="id")
    return Descriptor(path=Path(name), name=name, source=source, model="x", seconds=0,
                      description=d)  # fmt: skip


def _cat(name: str, tier: str = "important") -> dict:
    return {"name": name, "description": GOOD, "tier": tier}


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


def test_agent_designs_categories_and_fixes_rejected_proposal(tmp_path):
    descs = [_desc("a.pdf", "struk"), _desc("b.pdf", "struk"), _desc("c.jpg", "meme")]
    clusters = [Cluster(id=0, members=["a.pdf", "b.pdf"]), Cluster(id=1, members=["c.jpg"])]
    short = {"name": "Hiburan", "description": "meme", "tier": "temporary"}
    model = ScriptedModel([
        ("inspect_cluster", {"cluster_id": 0}),
        ("propose", {"categories": [_cat("Keuangan"), short]}),                # deskripsi pendek
        ("propose", {"categories": [_cat("Keuangan"), _cat("Hiburan", "temporary")]}),
    ])  # fmt: skip
    cfg = Settings(min_categories=2)
    ws = Workspace(clusters, descs, cfg)

    steps = run_agent(ws, model, cfg, Trace(tmp_path / "t.jsonl"))

    assert steps == 3
    assert [(c.name, c.tier, c.items) for c in ws.final] == [
        ("Keuangan", "important", []), ("Hiburan", "temporary", [])]  # isi diisi oleh assign
    last_tool_msg = model.seen[2][-1]
    assert last_tool_msg["role"] == "tool" and "terlalu pendek" in last_tool_msg["content"]


def test_repeated_failures_are_not_rerun_and_agent_is_reminded(tmp_path):
    ws = Workspace([Cluster(id=0, members=["a.pdf"])], [_desc("a.pdf", "struk")],
                   Settings(min_categories=1))  # fmt: skip
    bad = ("inspect_cluster", {"cluster_id": 9})  # tidak ada
    model = ScriptedModel([bad, bad, bad, ("propose", {"categories": [_cat("Keuangan")]})])
    events = []

    run_agent(ws, model, Settings(min_categories=1), Trace(tmp_path / "t.jsonl"),
              lambda kind, data: events.append(kind))  # fmt: skip

    assert "nomor yang ada: [0]" in model.seen[1][-1]["content"]
    assert "jangan diulang" in model.seen[2][-1]["content"]
    assert "stuck" in events and "Cluster saat ini" in model.seen[3][-1]["content"]
    assert ws.final is not None


def test_propose_warns_once_about_big_uninspected_clusters():
    descs = [_desc(f"f{i}.pdf", "struk") for i in range(12)]
    ws = Workspace([Cluster(id=0, members=[d.name for d in descs])], descs,
                   Settings(min_categories=1))  # fmt: skip

    first = ws.run("propose", {"categories": [_cat("Keuangan")]})
    assert first["diterima"] is False
    assert "belum diperiksa: [0]" in first["kesalahan"][0]
    second = ws.run("propose", {"categories": [_cat("Keuangan")]})  # agen tetap pada keputusannya
    assert second["diterima"] is True and second["peringatan"]


def test_propose_rejects_bad_names():
    ws = Workspace([Cluster(id=0, members=["a.pdf"])], [_desc("a.pdf", "struk")],
                   Settings(min_categories=1))  # fmt: skip
    errors = ws.run("propose", {"categories": [_cat("Keuangan"), _cat("Keuangan"),
                                               _cat(NONE)]})["kesalahan"]  # fmt: skip
    assert any("dipakai dua kali" in e for e in errors)
    assert any(NONE in e for e in errors)


def test_too_many_calls_in_one_turn_are_skipped(tmp_path):
    descs = [_desc(f"f{i}.pdf", "struk") for i in range(8)]
    ws = Workspace([Cluster(id=i, members=[f"f{i}.pdf"]) for i in range(8)], descs,
                   Settings(min_categories=1))  # fmt: skip

    class Greedy:
        name = "rakus"

        def __init__(self):
            self.turn = 0

        def chat(self, messages, tools, num_ctx):
            self.turn += 1
            if self.turn == 1:  # 8 inspect sekaligus
                calls = [{"function": {"name": "inspect_cluster", "arguments": {"cluster_id": i}}}
                         for i in range(8)]  # fmt: skip
            else:
                calls = [{"function": {"name": "propose",
                                       "arguments": {"categories": [_cat("Struk")]}}}]  # fmt: skip
            return {"role": "assistant", "content": "", "tool_calls": calls}

    events = []
    run_agent(ws, Greedy(), Settings(min_categories=1, agent_max_calls=6),
              Trace(tmp_path / "t.jsonl"), lambda k, d: events.append(d))  # fmt: skip

    results = [d["result"] for d in events if "result" in d]
    assert sum("dilewati" in r.get("error", "") for r in results) == 2
    assert ws.final is not None


def test_cluster_count_is_capped():
    class RandomEmbedder:
        def embed(self, model, texts):
            rng = np.random.default_rng(0)
            return rng.normal(size=(len(texts), 8)).tolist()

    descs = [_desc(f"f{i}.pdf", "dokumen") for i in range(200)]
    clusters = cluster_descriptors(descs, RandomEmbedder(), Settings(max_clusters=24))
    assert len(clusters) == 24  # 200 / 4 = 50, dibatasi supaya muat di konteks agen
    assert sum(len(c.members) for c in clusters) == 200


class FakeChooser:
    """Menjawab sesuai jenis file di prompt; untuk "ragu" jawabannya ikut urutan kategori."""

    def __init__(self):
        self.calls = []

    def choose(self, prompt, choices):
        self.calls.append((prompt, choices))
        if "Jenis: struk" in prompt:
            return "Keuangan"
        if "Jenis: meme" in prompt:
            return choices[0]  # berubah saat urutan dibalik
        if "Jenis: rusak" in prompt:
            raise ModelError("jawaban tidak sesuai skema")
        return NONE


def test_assign_sorts_each_file_twice_and_holds_doubtful_ones():
    cats = [Category(name="Keuangan", description=GOOD, tier="important"),
            Category(name="Hiburan", description="Meme dan gambar lucu dari internet",
                     tier="temporary")]  # fmt: skip
    descs = [_desc("a.pdf", "struk"), _desc("lucu.jpg", "meme", source="image"),
             _desc("x.txt", "catatan"), _desc("y.pdf", "rusak")]  # fmt: skip
    model = FakeChooser()

    assigned, review = assign(descs, cats, model)

    assert assigned == {"a.pdf": "Keuangan"}
    assert review == {"lucu.jpg": "ragu antara Keuangan dan Hiburan",
                      "x.txt": "tidak cocok ke kategori mana pun",
                      "y.pdf": "gagal dipilah: jawaban tidak sesuai skema"}  # fmt: skip
    (p1, c1), (p2, c2) = model.calls[0], model.calls[1]
    assert c1 == ["Keuangan", "Hiburan", NONE] and c2 == ["Hiburan", "Keuangan", NONE]
    assert p1.index("- Keuangan") < p1.index("- Hiburan") and p2.index("- Hiburan") < p2.index("- Keuangan")
    image_prompt = model.calls[2][0]
    assert "Nama file" not in image_prompt  # gambar dinilai tanpa nama file, seperti di M2
    assert "Nama file: a.pdf" in p1


def test_triage_categories_and_copies_are_added(sample):
    items = list_items(sample, CFG)
    _, copies = sample_with_copies(items, 400, CFG)
    agent_cats = build_categories([], items, copies)
    names = {c.name: c for c in agent_cats}
    assert "Visual Studio Code-2.app" in names["Aplikasi & Installer"].items
    assert names["Project"].tier == "important"
    assert "PPE-0" in names["Dataset"].items
    assert "musik.mp3" in names["Audio"].items

    invoice = Category(name="Tagihan", description="x" * 20, tier="important",
                       items=["Invoice_Maret.pdf"])  # fmt: skip
    cats = build_categories([invoice], items, copies)
    assert "Invoice_Maret-2.pdf" in cats[0].items  # salinan ikut kategori file aslinya


# def test_no_think_is_sent_to_ollama_only_when_asked():
#     from sorter.llm import Ollama

#     sent = []
#     llm = Ollama("http://127.0.0.1:11434", "qwen3:14b", 5)
#     llm._post = lambda path, body: sent.append(body) or {"message": {"content": ""}}
#     llm.chat([], [], 1024)
#     llm.think = False
#     llm.chat([], [], 1024)
#     assert "think" not in sent[0] and sent[1]["think"] is False