"""Agen taksonomi: Qwen menyusun kategori dari cluster dengan memakai tool.

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

SYSTEM = """Kamu agen yang menyusun folder kategori untuk merapikan folder Downloads satu orang.
Kamu menerima cluster hasil pengelompokan otomatis. Clusternya sengaja kecil-kecil.

Tugasmu:
1. Cluster bertanda CAMPURAN wajib diperiksa dengan inspect_cluster. Pindahkan item yang
   tidak cocok dengan split (isi items dengan nama file persis, bukan jenisnya). Item yang
   dipindah menjadi cluster baru; cluster lama tetap ada dengan nomor yang sama.
2. Pakai peek_item kalau ringkasan belum cukup untuk memutuskan.
3. Akhiri dengan propose: {min_c} sampai {max_c} kategori. Tidak perlu menggabungkan cluster
   dulu: satu kategori boleh berisi banyak cluster_ids.

Pakai nomor dari "cluster_aktif" di hasil tool terakhir. Jangan mengulang panggilan yang gagal.


Aturan propose:
- Setiap cluster yang masih ada masuk ke tepat satu kategori.
- name: nama folder pendek bahasa Indonesia, misal "Struk & Tagihan", "Kuliah".
- description: ciri isi yang konkret, cukup untuk memilah file baru ke kategori ini.
- tier "important" untuk yang perlu disimpan lama (identitas, kontrak, invoice, tagihan,
  sertifikat, dokumen kuliah atau kerja); "temporary" untuk yang boleh dibuang (meme,
  screenshot sesaat, foto hiburan). Kalau ragu, pilih "important".
- Nilai screenshot dari isinya: screenshot bukti transfer, tiket, atau peta ikut kategori
  isinya, bukan otomatis masuk hiburan.
- Jangan buat kategori berisi 1 item; gabungkan ke kategori yang paling dekat.
- Kategori harus konsisten: CV, kode, atau data kerja tidak masuk "Kuliah" hanya karena
  clusternya sama.

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
    _fn("inspect_cluster", "Lihat semua item dalam satu cluster beserta ringkasannya.",
        {"cluster_id": INT}, ["cluster_id"]),
    _fn("peek_item", "Lihat ringkasan lengkap dan kata kunci satu item.",
        {"name": {"type": "string"}}, ["name"]),
    # _fn("merge", "Gabungkan beberapa cluster menjadi satu cluster baru.",
    #     {"cluster_ids": {"type": "array", "items": INT}}, ["cluster_ids"]),
    _fn("split", "Pindahkan item tertentu dari sebuah cluster ke cluster baru.",
        {"cluster_id": INT, "items": {"type": "array", "items": {"type": "string"}}},
        ["cluster_id", "items"]),
    _fn("propose", "Kirim kategori final. Akan ditolak dengan daftar kesalahan bila tidak valid.",
        {"categories": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "tier": {"type": "string", "enum": ["important", "temporary"]},
                "cluster_ids": {"type": "array", "items": INT},
            },
            "required": ["name", "description", "tier", "cluster_ids"],
        }}}, ["categories"]),
]  # fmt: skip


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict], num_ctx: int) -> dict: ...

class UnknownCluster(Exception):
    pass


class BadItems(Exception):
    pass

class Workspace:
    """Keadaan yang diubah oleh tool: cluster saat ini dan hasil akhir bila propose diterima."""

    def __init__(self, clusters: list[Cluster], descriptors: list[Descriptor], cfg: Settings):
        self.clusters = {c.id: list(c.members) for c in clusters}
        self.by_name = {d.name: d for d in descriptors}
        self.next_id = max(self.clusters, default=-1) + 1
        self.cfg = cfg
        self.final: list[Category] | None = None
        self.seen: set[str] = set()  # item yang sudah dilihat agen lewat inspect atau peek
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

    def active(self) -> dict[str, str]:
        """Daftar cluster yang masih ada, dikirim ke agen setelah split atau propose ditolak."""
        return {
            str(i): f"{len(m)} item: {self.kinds(m)}" + (" [CAMPURAN]" if self.mixed(m) else "")
            for i, m in sorted(self.clusters.items())
        }

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
            return {
                "error": f"cluster {exc.args[0]} tidak ada (sudah digabung atau dipecah)",
                "cluster_aktif": self.active(),
            }
        except BadItems as exc:
            return {"error": exc.args[0], "item_di_cluster": exc.args[1]}
        except KeyError as exc:
            return {"error": f"item {exc} tidak ada, pakai nama file persis"}
        except (TypeError, ValueError) as exc:
            return {"error": f"argumen tidak valid: {exc}"}

    def tool_inspect_cluster(self, cluster_id: int) -> dict:
        members = self._get(cluster_id)
        self.seen.update(members[:20])
        return {
            "cluster_id": cluster_id,
            "items": [
                {
                    "name": n,
                    "jenis": self.by_name[n].description.doc_type,
                    "ringkasan": self.by_name[n].description.summary,
                }
                for n in members[:20]
            ],  # fmt: skip
            "lainnya": max(0, len(members) - 20),
        }

    def tool_peek_item(self, name: str) -> dict:
        d = self.by_name[name]
        self.seen.add(name)
        return {"name": name, "dilihat_sebagai": d.source, **d.description.model_dump()}

    # def tool_merge(self, cluster_ids: list[int]) -> dict:
    #     ids = [int(i) for i in cluster_ids]
    #     if len(ids) < 2:
    #         raise ValueError("merge butuh minimal 2 cluster")
    #     members = [n for i in ids for n in self._get(i)]
    #     for i in ids:
    #         del self.clusters[i]
    #     new_id = self._add(members)
    #     return {
    #         "cluster_baru": new_id,
    #         "jumlah_item": len(members),
    #         "isi": self.kinds(members),
    #         "sisa_cluster": len(self.clusters),
    #         "cluster_aktif": self.active(),
    #     }

    def tool_split(self, cluster_id: int, items: list[str]) -> dict:
        source = self._get(cluster_id)
        unknown = [n for n in items if n not in source]
        if unknown:
            raise BadItems(f"bukan nama file di cluster {cluster_id}: {unknown[:5]}", source)
        moving = [n for n in source if n in items]
        if not moving:
            raise BadItems("items kosong, isi dengan nama file yang mau dipindah", source)
        if len(moving) == len(source):
            raise BadItems(
                f"itu semua isi cluster {cluster_id}; tidak perlu split, cluster ini sudah "
                "terpisah. Masukkan langsung ke kategori yang cocok lewat propose",
                source,
            )
        self.clusters[int(cluster_id)] = [n for n in source if n not in moving]
        new_id = self._add(moving)
        return {
            "cluster_baru": new_id,
            "dipindah": moving,
            "isi": self.kinds(moving),
            "sisa_di_cluster_lama": len(source) - len(moving),
            "cluster_aktif": self.active(),
        }

    def tool_propose(self, categories: list[dict]) -> dict:
        """Kesalahan (errors) selalu menolak. Peringatan (warnings) hanya menolak sekali:
        kalau agen tetap mengusulkannya, dianggap keputusan sadar dan diterima."""
        errors, warnings = [], []
        low = min(self.cfg.min_categories, len(self.clusters))
        if not low <= len(categories) <= self.cfg.max_categories:
            errors.append(f"jumlah kategori harus {low} sampai {self.cfg.max_categories}")
        seen: dict[int, str] = {}
        for c in categories:
            if len(c.get("description", "")) < 20:
                errors.append(f"deskripsi '{c.get('name')}' terlalu pendek, buat lebih konkret")
            for i in c.get("cluster_ids", []):
                if int(i) not in self.clusters:
                    errors.append(f"cluster {i} tidak ada (mungkin sudah digabung)")
                elif int(i) in seen:
                    errors.append(f"cluster {i} dipakai di '{seen[int(i)]}' dan '{c['name']}'")
                seen[int(i)] = c.get("name", "")
            members = [n for i in c.get("cluster_ids", []) if int(i) in self.clusters
                       for n in self.clusters[int(i)]]  # fmt: skip
            if len(members) == 1 and len(categories) > low:
                warnings.append(f"'{c.get('name')}' hanya 1 item, sebaiknya gabung ke kategori lain")
        for i, m in self.clusters.items():
            if self.mixed(m) and not set(m) <= self.seen:
                warnings.append(f"cluster {i} campuran dan belum diperiksa (inspect_cluster)")
        missing = sorted(set(self.clusters) - set(seen))

        missing = sorted(set(self.clusters) - set(seen))
        if missing:
            errors.append(f"cluster belum masuk kategori mana pun: {missing}")
        new_warnings = [w for w in warnings if w not in self.warned]
        self.warned.update(warnings)
        if errors or new_warnings:
            return {
                "diterima": False,
                "kesalahan": errors + new_warnings,
                "cluster_aktif": self.active(),
            }
        self.final = [
            Category(
                name=c["name"],
                description=c["description"],
                tier=c["tier"],
                items=[n for i in c["cluster_ids"] for n in self.clusters[int(i)]],
            )
            for c in categories
        ]
        return {"diterima": True, "peringatan": warnings} if warnings else {"diterima": True}

    def _add(self, members: list[str]) -> int:
        new_id = self.next_id
        self.clusters[new_id] = members
        self.next_id += 1
        return new_id


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
    
    system = SYSTEM.format(min_c=cfg.min_categories, max_c=cfg.max_categories)
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
            messages.append({"role": "user", "content": "Panggil salah satu tool. Kalau sudah "
                             "yakin, panggil propose."})  # fmt: skip
            continue
        ok = False
        for call in calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            key = f"{name} {json.dumps(args, sort_keys=True)}"
            if key in failed:
                result = {"error": "panggilan yang sama persis sudah gagal sebelumnya, jangan "
                          "diulang. Lakukan hal lain atau propose"}  # fmt: skip
            else:
                result = workspace.run(name, args)
            if "error" in result:
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
                             "sekarang dengan nomor cluster di atas."})  # fmt: skip

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


def test_tool_errors_tell_the_agent_what_is_valid():
    descs = [_desc("a.pdf", "struk"), _desc("b.pdf", "struk"), _desc("c.jpg", "meme"),
             _desc("d.txt", "catatan"), _desc("e.py", "kode")]  # fmt: skip
    clusters = [Cluster(id=0, members=["a.pdf"]), Cluster(id=1, members=["b.pdf"]),
                Cluster(id=2, members=["c.jpg", "d.txt", "e.py"])]  # fmt: skip
    ws = Workspace(clusters, descs, Settings(min_categories=2))

    merged = ws.run("merge", {"cluster_ids": [0, 1]})  # cluster 0 dan 1 hilang, jadi cluster 3
    assert set(merged["cluster_aktif"]) == {"2", "3"}
    stale = ws.run("split", {"cluster_id": 0, "items": ["a.pdf"]})  # nomor lama
    assert "tidak ada" in stale["error"] and set(stale["cluster_aktif"]) == {"2", "3"}
    wrong = ws.run("split", {"cluster_id": 2, "items": ["kode"]})  # jenis, bukan nama file
    assert wrong["item_di_cluster"] == ["c.jpg", "d.txt", "e.py"]

    cats = [{"name": "Keuangan", "description": "Struk belanja dan bukti bayar harian",
             "tier": "important", "cluster_ids": [3]},
            {"name": "Lain", "description": "Meme, catatan, dan potongan kode",
             "tier": "temporary", "cluster_ids": [2]}]  # fmt: skip
    rejected = ws.run("propose", {"categories": cats})
    assert "cluster 2 campuran dan belum diperiksa" in " ".join(rejected["kesalahan"])
    ws.run("inspect_cluster", {"cluster_id": 2})
    assert ws.run("propose", {"categories": cats}) == {"diterima": True}


def test_propose_rejects_one_item_category():
    descs = [_desc(n, "struk") for n in ["a", "b", "c", "d"]]
    ws = Workspace([Cluster(id=i, members=[n]) for i, n in enumerate("abcd")], descs,
                   Settings(min_categories=2))  # fmt: skip
    cat = {"description": "Struk belanja dan bukti bayar harian", "tier": "important"}
    cats = [cat | {"name": "A", "cluster_ids": [0, 1]}, cat | {"name": "B", "cluster_ids": [2]},
            cat | {"name": "C", "cluster_ids": [3]}]  # fmt: skip
    errors = ws.run("propose", {"categories": cats})["kesalahan"]
    assert any("'B' hanya 1 item" in e for e in errors)