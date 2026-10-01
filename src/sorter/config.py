"""Semua aturan dan angka di satu tempat. Nilai milestone berikutnya ditambah saat dibutuhkan."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Output tiap run: runs/<run-id>/inventory.json
    runs_dir: Path = field(default_factory=lambda: Path(os.environ.get("SORTER_RUNS_DIR", "runs")))

    # Dilewati
    min_age_seconds: int = 10  # lebih baru dari ini mungkin masih ditulis
    partial_suffixes: tuple[str, ...] = (".crdownload", ".part", ".download", ".partial")
    ignored_names: tuple[str, ...] = (".DS_Store", ".localized", "Icon\r")

    # Folder yang di macOS tampil sebagai satu file. Isinya tidak pernah dibaca.
    package_suffixes: tuple[str, ...] = (
        ".app", ".bundle", ".framework", ".plugin", ".photoslibrary",
        ".pages", ".numbers", ".key", ".rtfd",
    )  # fmt: skip

    # Triage dari akhiran
    installer_suffixes: tuple[str, ...] = (".dmg", ".pkg", ".mpkg", ".iso", ".ipa", ".apk")

    # Media dan file teknis: jenisnya jelas dari akhiran, thumbnail tidak membantu model
    video_suffixes: tuple[str, ...] = (".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm")
    audio_suffixes: tuple[str, ...] = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".aiff")
    font_suffixes: tuple[str, ...] = (".otf", ".ttf", ".woff", ".woff2")
    model_suffixes: tuple[str, ...] = (
        ".pt", ".pth", ".onnx", ".safetensors", ".mlmodel", ".mlkitmodel", ".h5", ".ckpt",
    )  # fmt: skip

    archive_suffixes: tuple[str, ...] = (
        ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".gz", ".7z", ".rar",
    )  # fmt: skip

    # Triage folder: cukup satu penanda / satu pola
    project_markers: tuple[str, ...] = (
        ".git", "requirements.txt", "pyproject.toml", "setup.py", "package.json",
        "Cargo.toml", "go.mod", "Package.swift", "Podfile", "pom.xml", "build.gradle",
    )  # fmt: skip
    dataset_markers: tuple[str, ...] = ("data.yaml", "dataset.yaml", "classes.txt")
    dataset_dir_patterns: tuple[tuple[str, ...], ...] = (
        ("images", "labels"),
        ("images", "annotations"),
        ("train", "val"),
        ("train", "valid"),
        ("train", "test"),
    )

    # Pola nama salinan: "nama-2", "nama copy", "nama copy 2", "nama (1)"
    duplicate_name_patterns: tuple[str, ...] = (
        r"^(?P<base>.+)-\d{1,2}$",
        r"^(?P<base>.+) copy(?: \d)?$",
        r"^(?P<base>.+) \(\d{1,2}\)$",
    )
    quick_hash_bytes: int = 64 * 1024  # saringan awal sebelum hash penuh
    folder_signature_limit: int = 20000  # batas file yang didaftar per folder

    # M2: describe. Model hanya lewat Ollama di Mac ini; host selain localhost ditolak.
    ollama_host: str = field(
        default_factory=lambda: os.environ.get("SORTER_OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("SORTER_MODEL", "qwen3-vl:8b-instruct")
    )
    request_timeout_s: float = 180
    agent_timeout_s: float = 600  # model yang berpikir dulu bisa lama per giliran
    sample_size: int = 400
    cache_db: Path = field(
        default_factory=lambda: Path(os.environ.get("SORTER_CACHE_DB", "runs/cache.sqlite"))
    )

    # Cuplikan yang dikirim ke model
    text_max_chars: int = 2000
    image_max_px: int = 768
    folder_listing_max: int = 40
    image_suffixes: tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".webp", ".avif",
    )  # fmt: skip
    # Format yang punya pratinjau tapi tidak bisa dibaca teksnya: thumbnail QuickLook, khusus macOS
    quicklook: bool = field(
        default_factory=lambda: sys.platform == "darwin"
        and os.environ.get("SORTER_QUICKLOOK", "1") == "1"
    )
    quicklook_suffixes: tuple[str, ...] = (
        ".ppt", ".pps", ".xls", ".key", ".pages", ".numbers", ".svg", ".psd", ".eps",
    )  # fmt: skip
    text_suffixes: tuple[str, ...] = (
        ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".html", ".yaml", ".yml", ".log",
        ".py", ".js", ".ts", ".swift", ".sh", ".sql", ".ipynb",
        ".htm", ".aspx", ".ics", ".drawio", ".tex",
    )  # fmt: skip


    # M3: discover. Embedding lewat Ollama; agen memakai model yang sama dengan describe.
    # bge-m3 multibahasa; nomic-embed-text lebih kecil tapi lemah untuk bahasa Indonesia.
    embed_model: str = field(
        default_factory=lambda: os.environ.get("SORTER_EMBED_MODEL", "bge-m3")
    )
    min_categories: int = 3
    max_categories: int = 12
    items_per_cluster: int = 4  # cluster awal sengaja kecil-kecil; agen yang menggabungkan
    max_clusters: int = 24  # lebih dari ini, percakapan agen tidak muat di konteks model 8B
    agent_max_calls: int = 6  # panggilan tool per giliran; sisanya ditolak supaya konteks tidak meledak
    agent_max_steps: int = 30
    agent_num_ctx: int = 32768  # percakapan agen panjang; 16k terbukti kurang di Downloads asli
    inspect_items: int = 12  # item yang ditampilkan per inspect_cluster, supaya konteks tidak penuh