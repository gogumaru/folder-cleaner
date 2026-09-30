"""Rencana pemindahan (M4): setiap item mau dipindah ke mana, dan kenapa.

Murni data: file ini tidak menyentuh disk. Hasilnya plan.json, yang boleh diperiksa dan
diubah user sebelum apply.
"""

from pathlib import Path

from sorter.schemas import Descriptor, Item, Plan, PlanItem, Taxonomy, Triage

# Dua tingkat penyimpanan (PRD): penting ke Documents, sementara tetap di Downloads
TIER_ROOT = {"important": "Documents", "temporary": "Downloads"}


def folder_name(category: str) -> str:
    """Nama kategori jadi nama folder yang aman: tanpa garis miring, tidak "." atau ".."."""
    name = category.replace("/", "-").replace(":", "-").strip() or "Tanpa Nama"
    return "Kategori" if name in (".", "..") else name


def build_plan(
    taxonomy: Taxonomy, items: list[Item], descriptors: list[Descriptor], copies: dict[Path, Path]
) -> Plan:
    category_of = {n: c for c in taxonomy.categories for n in c.items}
    described = {d.name: d for d in descriptors if d.description}
    name_of = {it.path: it.name for it in items}
    rows = []
    for it in sorted(items, key=lambda i: i.name.lower()):
        original = name_of.get(copies.get(it.path))
        c = category_of.get(it.name)
        if it.triage is Triage.SKIPPED:
            rows.append(PlanItem(name=it.name, action="stay", reason=it.reason))
        elif c is None:
            rows.append(PlanItem(name=it.name, action="stay", reason=_why_unplaced(it, taxonomy,
                                                                                  described)))  # fmt: skip
        else:
            if c.source == "triage":
                source, reason = "aturan", it.reason + (f"; salinan identik {original}"
                                                       if original else "")  # fmt: skip
            elif original:
                source, reason = "salinan", f"salinan identik {original}; disarankan dihapus"
            else:
                d = described[it.name].description
                source, reason = "agen", f"{d.doc_type}: {d.summary}"
            rows.append(PlanItem(
                name=it.name, action="move", dest=f"{TIER_ROOT[c.tier]}/{folder_name(c.name)}",
                category=c.name, source=source, reason=reason, duplicate_of=original,
            ))  # fmt: skip
    return Plan(run_id=taxonomy.run_id, folder=taxonomy.folder, items=rows)


def _why_unplaced(it: Item, taxonomy: Taxonomy, described: dict) -> str:
    if it.name in taxonomy.unread:
        return "isinya tidak terbaca, tetap di tempat"
    if it.name not in described:
        return "belum dibaca model (di luar sampel), tetap di tempat"
    return "tidak masuk kategori mana pun, tetap di tempat"