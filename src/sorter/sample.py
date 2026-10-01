"""Folder contoh yang meniru pola Downloads asli, untuk mencoba sorter tanpa menyentuh
folder pribadi. Semua file berisi data palsu berukuran kecil.
"""

from __future__ import annotations

import os
import io
import time
import zipfile
from functools import lru_cache

from pathlib import Path

from annotated_types import doc

import pymupdf

PROTECTED = ("Downloads", "Documents", "Desktop", "Library", "Pictures", "Movies", "Music")
DAY = 86400


class SampleFolderError(Exception):
    pass


def _check_target(target: Path) -> Path:
    target = target.expanduser().resolve()
    home = Path.home().resolve()
    for name in PROTECTED:
        protected = home / name
        if target == protected or target.is_relative_to(protected):
            raise SampleFolderError(
                f"{target} ada di dalam ~/{name}. Folder contoh tidak boleh dibuat di sana."
            )
    if target.exists() and any(target.iterdir()):
        raise SampleFolderError(f"{target} sudah ada dan tidak kosong.")
    return target

@lru_cache
def _pdf(text: str) -> bytes:
    doc = pymupdf.open()
    doc.new_page(width=420, height=595).insert_text((40, 60), text, fontsize=12)
    return doc.tobytes(no_new_id=True)

@lru_cache
def _scan_pdf(text: str) -> bytes:
    """PDF hasil scan: halamannya hanya gambar, tanpa lapisan teks."""
    doc = pymupdf.open()
    doc.new_page(width=420, height=595).insert_image(
        pymupdf.Rect(20, 20, 400, 300), stream=_jpg(text)
    )
    return doc.tobytes(no_new_id=True)

@lru_cache
def _image(text: str, fmt: str, bg: tuple = (1, 1, 0.9)) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=260)
    page.draw_rect(page.rect, color=None, fill=bg)
    page.insert_text((24, 50), text, fontsize=16)
    return page.get_pixmap(dpi=96).tobytes(fmt)


def _jpg(text: str) -> bytes:
    return _image(text, "jpg")


def _png(text: str) -> bytes:
    return _image(text, "png", (0.93, 0.94, 0.96))


@lru_cache
def _docx(text: str) -> bytes:
    paragraphs = "".join(f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>" for line in text.split("\n"))
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        fixed = (2026, 1, 1, 0, 0, 0)  # tanggal tetap supaya byte-nya stabil
        z.writestr(zipfile.ZipInfo("[Content_Types].xml", fixed), "<Types/>")
        z.writestr(
            zipfile.ZipInfo("word/document.xml", fixed),
            f"<w:document {ns}><w:body>{paragraphs}</w:body></w:document>",
        )
    return buf.getvalue()

def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    path.write_bytes(data)


def _yolo_dataset(root: Path, n: int, tag: str) -> None:
    _write(root / "data.yaml", f"train: train/images\nval: valid/images\nnc: 3\n# {tag}\n")
    for split in ("train", "valid", "test"):
        for i in range(n):
            _write(root / split / "images" / f"img_{i:04d}.jpg", _jpg(f"{tag}{split}{i}"))
            _write(root / split / "labels" / f"img_{i:04d}.txt", f"0 0.5 0.5 0.1 0.1 {tag}\n")


def _app(root: Path, version: str) -> None:
    _write(root / "Contents" / "Info.plist", f"<plist><string>{version}</string></plist>")
    _write(root / "Contents" / "MacOS" / "Electron", b"\x00binary-palsu" * 20)
    _write(root / "Contents" / "Resources" / "app.icns", b"icns-palsu")


def _set_times(root: Path, age_days: float) -> None:
    ts = time.time() - age_days * DAY
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames + dirnames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.utime(p, (ts, ts))
        os.utime(dirpath, (ts, ts))

STRUK = """TOKO SEJAHTERA
Jl. Melati 12, Bandung

Beras 5kg      68.000
Telur 1kg      28.500
TOTAL          96.500"""
INVOICE = """INVOICE No. INV-2026-03
Kepada: PT Contoh Maju
Jasa desain aplikasi mobile
Total: Rp 4.500.000
Jatuh tempo: 31 Maret 2026"""
TIKET = """E-TICKET / BOARDING PASS
Maskapai Nusantara Air NA-718
Jakarta (CGK) -> Denpasar (DPS)
14 Oktober 2026, 07:25
Penumpang: A. Pratama"""
LAPORAN = "Laporan Kegiatan Bakti Sosial\nTanggal: 3 Agustus 2026\nPeserta: 24 orang"
LAPORAN_REVISI = LAPORAN.replace("24", "27") + "\nAnggaran: Rp 2.100.000"
MEME = "Ketika kodenya jalan\ndi percobaan pertama:\n\n(wajah curiga)"
SCREENSHOT = "System Settings > Wi-Fi\nJaringan: KostMelati_5G\nStatus: Terhubung"


# M3: item tambahan supaya clustering punya bahan. Nama sengaja campur: sebagian tidak informatif.
# (nama file, jenis pembuat, isi)
# fmt: off
MORE = [
    # Struk dan bukti bayar
    ("scan0002.pdf", "scan", "MINIMARKET CERIA\nSusu UHT 1L   18.900\nRoti tawar    15.500\nTOTAL         34.400"),
    ("scan0003.pdf", "scan", "APOTEK SEHAT SENTOSA\nParacetamol 500mg  12.000\nVitamin C          25.000\nTOTAL              37.000"),
    ("IMG_20260812_101500.jpg", "jpg", "KEDAI KOPI SENJA\nKopi susu gula aren x2\nTOTAL Rp 44.000\nTerima kasih!"),
    ("struk.pdf", "scan", "SPBU 34.401.02\nPertalite 20,5 L\nTOTAL Rp 205.000"),
    ("Screenshot 2026-08-20 at 19.02.11.png", "png", "Transfer Berhasil\nKe: Kos Melati\nRp 1.500.000\nBerita: sewa September"),
    # Tagihan dan invoice
    ("tagihan-agustus.pdf", "pdf", "TAGIHAN LISTRIK\nPeriode: Agustus 2026\nID Pelanggan: 5123 4567 890\nTotal tagihan: Rp 312.450\nJatuh tempo: 20 September 2026"),
    ("download.pdf", "pdf", "TAGIHAN INTERNET RUMAH\nPaket 50 Mbps\nBulan: September 2026\nTotal: Rp 385.000"),
    ("INV-0042.pdf", "pdf", "INVOICE No. INV-0042\nLayanan hosting tahunan\nKepada: Studio Kreatif Nusa\nTotal: Rp 1.200.000"),
    ("kwitansi.pdf", "scan", "KWITANSI\nTelah diterima dari: A. Pratama\nUntuk pembayaran: sewa kos Oktober\nSebesar: Rp 1.500.000"),
    # Tiket
    ("e-ticket-kereta.pdf", "pdf", "E-TIKET KERETA API\nKA Parahyangan Ekspres\nBandung -> Gambir\n2 Oktober 2026, 05:10\nKursi: EKS-3 / 12A"),
    ("tiket (2).pdf", "pdf", "TIKET BIOSKOP\nFilm: Laskar Senja\nStudio 3, Kursi F7\nSabtu, 27 September 2026, 19:15"),
    ("booking.pdf", "pdf", "KONFIRMASI HOTEL\nHotel Pantai Indah, Denpasar\nCheck-in 14 Okt 2026, check-out 17 Okt 2026\n1 kamar deluxe"),
    ("boarding-pass-pulang.pdf", "pdf", "E-TICKET / BOARDING PASS\nMaskapai Nusantara Air NA-721\nDenpasar (DPS) -> Jakarta (CGK)\n17 Oktober 2026, 18:40"),
    # Kuliah
    ("Pertemuan5_BasisData.pdf", "pdf", "BASIS DATA - Pertemuan 5\nNormalisasi: 1NF, 2NF, 3NF\nContoh tabel mahasiswa dan mata kuliah"),
    ("jadwal.pdf", "pdf", "JADWAL UJIAN TENGAH SEMESTER\nSemester Ganjil 2026/2027\nSenin: Kalkulus II, Rabu: Basis Data, Jumat: Statistika"),
    ("KRS.pdf", "pdf", "KARTU RENCANA STUDI\nSemester Ganjil 2026/2027\nKalkulus II (3 SKS), Basis Data (3 SKS), Statistika (3 SKS)\nTotal: 21 SKS"),
    ("tugas_kalkulus_final.docx", "docx", "Tugas Kalkulus II\nSoal 1: Hitung integral lipat dua\nSoal 2: Deret Taylor"),
    ("silabus.pdf", "pdf", "SILABUS MATA KULIAH MACHINE LEARNING\nMinggu 1: regresi linier\nMinggu 2: klasifikasi\nMinggu 3: clustering"),
    ("document (3).pdf", "pdf", "TRANSKRIP NILAI SEMENTARA\nIPK: 3,61\nJumlah SKS lulus: 84"),
    # Dokumen pribadi penting
    ("sertifikat.pdf", "pdf", "SERTIFIKAT\nDiberikan kepada A. Pratama\nAtas penyelesaian Pelatihan Python untuk Analisis Data\n40 jam pelajaran"),
    ("surat-aktif.pdf", "pdf", "SURAT KETERANGAN AKTIF KULIAH\nMenerangkan bahwa A. Pratama\nadalah mahasiswa aktif semester 5"),
    ("kontrak_magang_signed.pdf", "scan", "PERJANJIAN MAGANG\nAntara PT Contoh Maju dan A. Pratama\nPeriode: Juli sampai Desember 2026"),
    ("CV_2026.docx", "docx", "CURRICULUM VITAE\nA. Pratama\nPendidikan: S1 Informatika\nKeahlian: Python, SQL, Figma"),
    # Screenshot
    ("Screenshot 2026-09-10 at 08.15.40.png", "png", "Google Maps\nRute: Kos Melati -> Kampus\n12 menit (4,1 km)"),
    ("Screenshot 2026-09-12 at 22.40.05.png", "png", "Terminal\nModuleNotFoundError: No module named 'pymupdf'"),
    ("Screenshot 2026-09-15 at 13.01.22.png", "png", "Grup Kelas Basis Data\nDosen: Kuis dimajukan ke hari Rabu ya"),
    # Meme dan foto (gambar bertulisan: model hanya bisa membaca tulisannya)
    ("images.jpg", "jpg", "Meme: 'Deadline besok'\nSaya: baru buka laptop"),
    ("IMG_7731.jpg", "jpg", "Meme: 'Hasil ekspektasi vs realita'\nkue ulang tahun yang miring"),
    ("download (1).jpg", "jpg", "Meme: kucing memakai kacamata\n'Mode serius: ON'"),
    ("IMG_6012.jpg", "jpg", "Foto: kucing oren tidur di sofa"),
    ("IMG_6013.jpg", "jpg", "Foto: nasi goreng dan es teh di warung"),
    ("IMG_6020.jpg", "jpg", "Foto: pemandangan gunung saat pagi berkabut"),
    # Kode, data, catatan
    ("analisis.py", "text", "import pandas as pd\n\ndf = pd.read_csv('penjualan.csv')\nprint(df.groupby('bulan').sum())\n"),
    ("penjualan.csv", "text", "bulan,produk,jumlah\nJanuari,Kopi,120\nFebruari,Kopi,98\nMaret,Teh,77\n"),
    ("config.json", "text", "{\"theme\": \"dark\", \"fontSize\": 14, \"autosave\": true}\n"),
    ("todo.txt", "text", "TODO minggu ini:\n- kumpul tugas kalkulus\n- bayar kos\n- servis motor\n"),
    ("ide-skripsi.md", "text", "# Ide skripsi\n- Klasifikasi dokumen otomatis dengan model lokal\n- Deteksi duplikat foto\n"),
    ("resep.txt", "text", "Resep nasi goreng: nasi, bawang, kecap, telur, cabai.\n"),
]

def _add_more(t: Path) -> None:
    makers = {"pdf": _pdf, "scan": _scan_pdf, "jpg": _jpg, "png": _png, "docx": _docx}
    for name, kind, text in MORE:
        _write(t / name, text if kind == "text" else makers[kind](text))


def make_sample_folder(target: Path) -> Path:
    """Buat folder contoh di `target`. Mengembalikan path absolutnya."""
    t = _check_target(target)
    t.mkdir(parents=True, exist_ok=True)

    # Dokumen dan gambar dengan nama tidak informatif (perlu dibaca model)
    _write(t / "scan0001.pdf", _pdf("Struk belanja Indomaret 12.500"))
    _write(t / "scan0001.pdf", _scan_pdf(STRUK))  # hasil scan: tanpa lapisan teks
    _write(t / "salinan-scan.pdf", _scan_pdf(STRUK))  # isi sama, nama beda
    _write(t / "Invoice_Maret.pdf", _pdf(INVOICE))
    _write(t / "Invoice_Maret-2.pdf", _pdf(INVOICE))
    _write(t / "tiket-pesawat.pdf", _pdf(TIKET))
    _write(t / "laporan.docx", _docx(LAPORAN))
    _write(t / "laporan copy.docx", _docx(LAPORAN_REVISI))  # nama mirip, isi beda
    _write(t / "IMG_4821.jpg", _image("Pantai saat matahari terbenam", "jpg", (1, 0.75, 0.5)))
    _write(t / "meme.jpg", _jpg(MEME))
    _write(t / "meme (1).jpg", _jpg(MEME))
    _write(t / "Screenshot 2026-09-01 at 10.12.33.png", _png(SCREENSHOT))
    _write(t / "catatan.txt", "Belanja minggu ini: telur, beras, kopi, sabun cuci\n")

    _write(t / "musik.mp3", b"ID3" + b"\x00" * 200)
    _write(t / "backup.dat", bytes(range(256)) * 4)  # format tak dikenal: hanya namanya

    # Installer dan arsip
    _write(t / "Docker.dmg", b"koly-palsu" * 50)
    _write(t / "python-3.13.pkg", b"xar!-palsu" * 50)
    _write(t / "dataset-backup.zip", b"PK\x03\x04" + b"\x00" * 100)
    _write(t / "logs.tar.gz", b"\x1f\x8b" + b"\x00" * 100)

    # Bundle aplikasi dan duplikatnya
    _app(t / "Visual Studio Code.app", "1.93")
    _app(t / "Visual Studio Code-2.app", "1.93")
    _app(t / "Visual Studio Code-3.app", "1.93")

    # Project
    proj = t / "yolov8-silva-main"
    _write(proj / "requirements.txt", "ultralytics\n")
    _write(proj / "train.py", "print('train')\n")
    _write(proj / ".git" / "HEAD", "ref: refs/heads/main\n")
    _yolo_dataset(proj / "datasets" / "silva", 3, "silva")
    # Project hasil unzip yang dibungkus satu folder lagi
    _write(t / "Face Detection" / "face-detection-main" / "package.json", '{"name": "face"}\n')
    _write(t / "Face Detection" / "face-detection-main" / "index.js", "console.log(1)\n")

    # Dataset
    _yolo_dataset(t / "PPE-0", 4, "ppe")
    _write(t / "coco" / "images" / "000001.jpg", _jpg("coco1"))
    _write(t / "coco" / "labels" / "000001.txt", "1 0.3 0.3 0.2 0.2\n")
    _write(t / "speechocean762" / "train" / "wav.scp", "utt1 WAVE/a.wav\n")
    _write(t / "speechocean762" / "test" / "wav.scp", "utt2 WAVE/b.wav\n")
    _write(t / "speechocean762" / "WAVE" / "a.wav", b"RIFF" + b"\x00" * 64)

    # Folder duplikat
    _yolo_dataset(t / "Innowork-24", 2, "inno")
    _yolo_dataset(t / "Innowork-24-2", 2, "inno")  # identik
    _yolo_dataset(t / "maskYOLOv5", 2, "mask")
    _yolo_dataset(t / "maskYOLOv5-2", 2, "mask")
    _write(t / "maskYOLOv5-2" / "catatan-tambahan.txt", "beda satu file\n")  # tidak identik

    # Folder biasa tanpa penanda
    _write(t / "Tugas Kuliah" / "tugas1.pdf", _pdf("Tugas 1 Statistika: rata-rata dan median"))
    _write(t / "Tugas Kuliah" / "tugas2.pdf", _pdf("Tugas 2 Statistika: uji hipotesis"))

    # Yang harus dilewati
    _write(t / "film.mp4.crdownload", b"\x00" * 64)
    _write(t / ".DS_Store", b"\x00" * 16)
    _write(t / ".kontrak.pdf.icloud", b"bplist-palsu")

    _add_more(t)
    _set_times(t, age_days=30)

    # Dibuat setelah _set_times: umurnya beberapa detik, jadi dilewati sebagai "terlalu baru"
    _write(t / "baru-diunduh.pdf", _pdf("Brosur promo kartu kredit"))
    os.symlink(t / "catatan.txt", t / "pintasan-catatan")
    return t