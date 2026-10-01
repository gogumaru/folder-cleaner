"""Agen taksonomi (hybrid): Qwen merancang kategori dari ringkasan cluster.

Agen hanya merancang nama, deskripsi, dan tingkat kategori. Ia tidak memindah file dan tidak
membagi cluster ke kategori. Setelah itu setiap file dipilah satu per satu oleh model
(assign.py) berdasarkan deskripsi itu, sama seperti watcher nanti.

Satu "giliran" agen: model menerima percakapan sejauh ini, lalu memilih tool yang dipanggil.
Kode menjalankan tool itu, mengirim hasilnya balik sebagai pesan "tool", dan model melanjutkan.
Loop berhenti ketika propose diterima, atau batas langkah habis.
"""

import json
import time
from collections import Counter
from collections.abc import Callable
from typing import Protocol

from sorter.cluster import cluster_overview, is_mixed
from sorter.config import Settings
from sorter.schemas import Category, Cluster, Descriptor, Item, Triage
from sorter.store import Trace

SYSTEM = """Kamu merancang folder kategori untuk merapikan folder Downloads seseorang.
Kamu menerima ringkasan cluster: kelompok file yang isinya mirip. Kamu TIDAK memindah file.
Setelah kamu selesai, model lain akan memilah setiap file satu per satu ke kategori yang
deskripsinya paling cocok. Jadi deskripsi yang kamu tulis adalah satu-satunya pegangannya.

Langkah:
1. Periksa cluster besar (10 item atau lebih) dengan inspect_cluster, dan peek_item bila perlu.
   Maksimal {max_calls} panggilan tool per giliran.
2. Akhiri dengan propose: {min_c} sampai {max_c} kategori.

Aturan kategori:
- Buat kategori yang cukup luas, idealnya 6 sampai 10. Jenis yang jarang (hanya 1 sampai 2
  file) masuk ke kategori luas yang paling dekat, bukan kategori sendiri.
- Kategori tidak boleh tumpang tindih: satu file hanya cocok ke satu kategori. Kalau dua
  kategori mirip, gabungkan, atau tulis batasnya dengan jelas di deskripsi.
- name: nama folder pendek bahasa Indonesia.
- description: 1-2 kalimat, ciri isi yang konkret dan apa bedanya dengan kategori lain.
  Jangan menyebut nama file tertentu.
- tier "important" untuk yang perlu disimpan lama (identitas, kontrak, keuangan, dokumen
  kuliah atau kerja, foto dan video pribadi); "temporary" untuk yang tidak perlu disimpan lama
  (meme, gambar dari internet, screenshot sesaat). Kalau ragu, pilih "important".
- Foto pribadi (kamera, orang, keluarga, acara, perjalanan) punya kategori sendiri dan selalu
  "important". Gambar dari internet (ilustrasi, logo, meme) dipisah darinya.

Selalu jawab dengan memanggil tool, bukan teks biasa."""


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


INT = {"type": "integer"}
TOOLS = [
    _fn("inspect_cluster", "Lihat jenis semua item dan contoh ringkasan dalam satu cluster.",
        {"cluster_id": INT}, ["cluster_id"]),
    _fn("peek_item", "Lihat ringkasan lengkap dan kata kunci satu item.",
        {"name": {"type": "string"}}, ["name"]),
    _fn("propose", "Kirim rancangan kategori. Akan ditolak dengan daftar kesalahan bila tidak valid.",
        {"categories": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "tier": {"type": "string", "enum": ["important", "temporary"]},
            },
            "required": ["name", "description", "tier"],
        }}}, ["categories"]),
]  # fmt: skip

NONE = "Tidak cocok"  # pilihan cadangan saat memilah; tidak boleh jadi nama kategori


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict], num_ctx: int) -> dict: ...


class UnknownCluster(Exception):
    pass


class Workspace:
    """Keadaan yang dilihat tool: cluster (tetap) dan rancangan akhir bila propose diterima."""

    def __init__(self, clusters: list[Cluster], descriptors: list[Descriptor], cfg: Settings):
        self.clusters = {c.id: list(c.members) for c in clusters}
        self.by_name = {d.name: d for d in descriptors}
        self.cfg = cfg
        self.final: list[Category] | None = None
        self.inspected: set[int] = set()  # cluster yang sudah dilihat agen
        self.warned: set[str] = set()  # peringatan propose yang sudah pernah disampaikan

    def overview(self) -> str:
        return "\n".join(
            cluster_overview(Cluster(id=i, members=m), self.by_name)
            for i, m in sorted(self.clusters.items())
        )

    def kinds(self, members: list[str]) -> str:
        """Jenis item yang paling sering dalam sekelompok item, misal "struk belanja x3"."""
        counts = Counter(self.by_name[n].description.doc_type for n in members)
        return ", ".join(f"{t} x{c}" for t, c in counts.most_common(3))

    def mixed(self, members: list[str]) -> bool:
        return is_mixed(Counter(self.by_name[n].description.doc_type for n in members))

    def _get(self, cluster_id) -> list[str]:
        cid = int(cluster_id)
        if cid not in self.clusters:
            raise UnknownCluster(cid)
        return self.clusters[cid]

    def run(self, name: str, args: dict) -> dict:
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return {"error": f"tool {name} tidak ada"}
        try:
            return handler(**args)
        except UnknownCluster as exc:
            return {"error": f"cluster {exc.args[0]} tidak ada; nomor yang ada: "
                             f"{sorted(self.clusters)}"}  # fmt: skip
        except KeyError as exc:
            return {"error": f"item {exc} tidak ada, pakai nama file persis"}
        except (TypeError, ValueError) as exc:
            return {"error": f"argumen tidak valid: {exc}"}

    def tool_inspect_cluster(self, cluster_id: int) -> dict:
        members = self._get(cluster_id)
        self.inspected.add(int(cluster_id))
        return {
            "cluster_id": cluster_id,
            "jenis_semua_item": ", ".join(f"{t} x{c}" for t, c in Counter(
                self.by_name[n].description.doc_type for n in members).most_common()),
            "items": [
                {
                    "name": n,
                    "jenis": self.by_name[n].description.doc_type,
                    "ringkasan": self.by_name[n].description.summary[:120],
                }
                for n in members[: self.cfg.inspect_items]
            ],
            "lainnya": max(0, len(members) - self.cfg.inspect_items),
        }  # fmt: skip

    def tool_peek_item(self, name: str) -> dict:
        d = self.by_name[name]
        return {"name": name, "dilihat_sebagai": d.source, **d.description.model_dump()}

    def tool_propose(self, categories: list[dict]) -> dict:
        """Kesalahan (errors) selalu menolak. Peringatan (warnings) hanya menolak sekali:
        kalau agen tetap mengusulkannya, dianggap keputusan sadar dan diterima."""
        errors, warnings = [], []
        if not self.cfg.min_categories <= len(categories) <= self.cfg.max_categories:
            errors.append(
                f"jumlah kategori harus {self.cfg.min_categories} sampai {self.cfg.max_categories}"
            )
        names = [c.get("name", "").strip() for c in categories]
        for c, name in zip(categories, names, strict=True):
            if not name or name.lower() == NONE.lower():
                errors.append(f"nama kategori tidak boleh kosong atau '{NONE}'")
            if len(c.get("description", "")) < 20:
                errors.append(f"deskripsi '{name}' terlalu pendek, buat lebih konkret")
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            errors.append(f"nama kategori dipakai dua kali: {dupes}")
        big = sorted(i for i, m in self.clusters.items()
                     if i not in self.inspected and len(m) >= 10)  # cluster kecil tidak wajib  # fmt: skip
        if big:
            warnings.append(f"cluster besar atau campuran belum diperiksa: {big}")
        new_warnings = [w for w in warnings if w not in self.warned]
        self.warned.update(warnings)
        if errors or new_warnings:
            return {"diterima": False, "kesalahan": errors + new_warnings}
        self.final = [
            Category(name=c["name"].strip(), description=c["description"], tier=c["tier"])
            for c in categories
        ]
        return {"diterima": True, "peringatan": warnings} if warnings else {"diterima": True}


def run_agent(
    workspace: Workspace,
    model: ChatModel,
    cfg: Settings,
    trace: Trace,
    on_event: Callable[[str, dict], None] | None = None,
    extra_note: str = "",
) -> int:
    """Jalankan loop agen. Mengembalikan jumlah giliran; hasilnya ada di workspace.final.

    on_event dipanggil untuk setiap kejadian supaya prosesnya bisa ditampilkan:
    "turn_start", "turn_end", "tool", "nudge", dan "stuck".
    """
    emit = on_event or (lambda kind, data: None)
    system = SYSTEM.format(min_c=cfg.min_categories, max_c=cfg.max_categories,
                           max_calls=cfg.agent_max_calls)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Cluster saat ini:\n{workspace.overview()}\n{extra_note}"},
    ]
    nudges = 0
    failed: set[str] = set()  # panggilan yang pernah gagal, supaya tidak diulang terus
    stuck = 0  # giliran berturut-turut yang semua panggilannya gagal
    for step in range(1, cfg.agent_max_steps + 1):
        emit("turn_start", {"step": step})
        t0 = time.perf_counter()
        reply = model.chat(messages, TOOLS, cfg.agent_num_ctx)
        seconds = round(time.perf_counter() - t0, 2)
        calls = reply.get("tool_calls") or []
        content = (reply.get("content") or "").strip()
        trace.log("agent_turn", str(step), content=content, tool_calls=calls, seconds=seconds)
        emit(
            "turn_end", {"step": step, "seconds": seconds, "content": content, "calls": len(calls)}
        )
        messages.append(reply)
        if not calls:
            nudges += 1
            emit("nudge", {"step": step, "count": nudges})
            if nudges > 3:
                break
            messages.append({"role": "user", "content": "Panggil salah satu tool, bukan teks. "
                             f"Cluster saat ini:\n{workspace.overview()}\nKalau sudah yakin, "
                             "panggil propose."})  # fmt: skip
            continue
        ok = False
        for i, call in enumerate(calls):
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            key = f"{name} {json.dumps(args, sort_keys=True)}"
            skipped = i >= cfg.agent_max_calls
            if skipped:
                result = {"error": f"dilewati: maksimal {cfg.agent_max_calls} panggilan per "
                          "giliran. Panggil lagi di giliran berikutnya bila masih perlu"}  # fmt: skip
            elif key in failed:
                result = {"error": "panggilan yang sama persis sudah gagal sebelumnya, jangan "
                          "diulang. Lakukan hal lain atau propose"}  # fmt: skip
            else:
                result = workspace.run(name, args)
            if "error" in result:
                if not skipped:
                    failed.add(key)  # propose yang ditolak tidak dicatat: boleh dikirim ulang
            elif result.get("diterima") is not False:
                ok = True
            trace.log("tool", name, args=args, result=result)
            emit("tool", {"step": step, "name": name, "args": args, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            if workspace.final is not None:
                return step
        stuck = 0 if ok else stuck + 1
        if stuck >= 3:
            stuck = 0
            emit("stuck", {"step": step})
            messages.append({"role": "user", "content": "Tiga giliran terakhir semuanya gagal. "
                             f"Cluster saat ini:\n{workspace.overview()}\nPanggil propose "
                             "sekarang."})  # fmt: skip
    return step


# Kategori tetap untuk item yang sudah jelas dari triage M1 (tidak lewat model)
_APPS = ("Aplikasi & Installer", "Bundle aplikasi .app dan installer .dmg/.pkg.", "temporary")
TRIAGE_CATEGORIES = {
    Triage.APP: _APPS,
    Triage.INSTALLER: _APPS,
    Triage.ARCHIVE: ("Arsip", "File arsip terkompresi seperti .zip dan .tar.gz.", "temporary"),
    Triage.PROJECT: ("Project", "Folder project kode: ada .git, package.json, dsb.", "important"),
    Triage.DATASET: (
        "Dataset",
        "Folder dataset: ada data.yaml atau images/ + labels/.",
        "temporary",
    ),
    # Video diperlakukan seperti foto pribadi: penting
    Triage.VIDEO: ("Video", "File video: rekaman kamera, screen recording, unduhan.", "important"),
    Triage.AUDIO: ("Audio", "File audio: musik, rekaman suara, efek suara.", "temporary"),
    Triage.FONT: ("Font", "File font .otf, .ttf, dan sejenisnya.", "temporary"),
    Triage.MODEL: ("Model ML", "Bobot model machine learning: .pt, .onnx, .safetensors.",
                   "temporary"),  # fmt: skip
}


def build_categories(
    agent_categories: list[Category], items: list[Item], copies: dict
) -> list[Category]:
    """Tambahkan salinan identik ke kategori file aslinya, lalu kategori tetap dari triage."""
    by_path_name = {it.path: it.name for it in items}
    result = [c.model_copy(deep=True) for c in agent_categories]
    for copy, original in copies.items():
        for c in result:
            if by_path_name.get(original) in c.items:
                c.items.append(by_path_name[copy])
    fixed: dict[str, Category] = {}
    for it in items:
        if it.triage in TRIAGE_CATEGORIES:
            name, desc, tier = TRIAGE_CATEGORIES[it.triage]
            fixed.setdefault(
                name, Category(name=name, description=desc, tier=tier, source="triage")
            )
            fixed[name].items.append(it.name)
    return result + list(fixed.values())