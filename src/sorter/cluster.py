"""Cluster: ubah ringkasan jadi embedding, lalu kelompokkan item yang mirip.

Cluster awal sengaja dibuat kecil-kecil (sekitar 4 item per cluster). Menggabungkan cluster
yang sejenis dan memberi nama adalah tugas agen, bukan tugas matematika di sini.
"""

from collections import Counter
from typing import Protocol

import numpy as np

from sorter import cluster
from sorter.config import Settings
from sorter.schemas import Cluster, Descriptor


class Embedder(Protocol):
    def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...


def embed_text(d: Descriptor) -> str:
    desc = d.description
    return f"{desc.doc_type}. {desc.summary} Kata kunci: {', '.join(desc.keywords)}"


def embed(descriptors: list[Descriptor], model: Embedder, cfg: Settings) -> np.ndarray:
    # nomic-embed-text bekerja paling baik dengan awalan tugas
    prefix = "clustering: " if cfg.embed_model.startswith("nomic") else ""
    texts = [prefix + embed_text(d) for d in descriptors]
    vectors = []
    for i in range(0, len(texts), 64):  # dikirim per 64 supaya request tidak terlalu besar
        vectors += model.embed(cfg.embed_model, texts[i : i + 64])
    x = np.array(vectors, dtype=float)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def agglomerate(x: np.ndarray, k: int) -> list[list[int]]:
    """Gabungkan dua kelompok paling mirip (rata-rata cosine) berulang kali sampai tersisa k."""
    n = len(x)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    members = [[i] for i in range(n)]
    alive = n
    while alive > k:
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        a, b = len(members[i]), len(members[j])
        row = (a * sim[i] + b * sim[j]) / (a + b)  # average linkage
        sim[i, :] = row
        sim[:, i] = row
        sim[i, i] = -np.inf
        sim[j, :] = -np.inf
        sim[:, j] = -np.inf
        members[i] += members[j]
        members[j] = []
        alive -= 1
    groups = [m for m in members if m]
    return sorted(groups, key=lambda g: (-len(g), min(g)))


def cluster_descriptors(
    descriptors: list[Descriptor], model: Embedder, cfg: Settings
) -> list[Cluster]:
    if not descriptors:
        return []
    k = max(1, min(len(descriptors), round(len(descriptors) / cfg.items_per_cluster)))
    groups = agglomerate(embed(descriptors, model, cfg), k)
    return [Cluster(id=i, members=[descriptors[j].name for j in g]) for i, g in enumerate(groups)]

def is_mixed(types: Counter) -> bool:
    """Cluster dianggap campuran bila isinya punya 3 jenis dokumen atau lebih."""
    return len(types) >= 3

def cluster_overview(cluster: Cluster, by_name: dict[str, Descriptor], examples: int = 3) -> str:
    """Satu baris ringkasan cluster untuk agen: semua jenis item + contoh ringkasan."""
    types = Counter(by_name[n].description.doc_type for n in cluster.members)
    kinds = ", ".join(f"{t} x{c}" for t, c in types.most_common())
    flag = " [CAMPURAN: periksa, split bila perlu]" if is_mixed(types) else ""
    samples = " | ".join(by_name[n].description.summary[:90] for n in cluster.members[:examples])
    return f"Cluster {cluster.id} ({len(cluster.members)} item){flag}: {kinds}. Contoh: {samples}"