# folder-cleaner

Prototype Python agen yang merapikan file berdasarkan isinya, sepenuhnya lokal.
Saat ini sampai M3: scan + triage + duplikat (M1), Qwen membaca isi item (M2), lalu agen Qwen
menyusun kategori (M3). Semuanya baca-saja: belum ada file yang dipindah.

## Mulai

```bash
uv sync
uv run sorter sample-folder ~/sorter-sample   # folder contoh berisi data palsu
uv run sorter scan ~/sorter-sample            # tambahkan --details untuk semua item
uv run pytest
```

Hasil scan ditulis ke `runs/<run-id>/inventory.json`.

## Describe (M2): Qwen membaca isi item

Butuh [Ollama](https://ollama.com) 0.12.7 atau lebih baru.

```bash
ollama pull qwen3-vl:8b-instruct 
uv run sorter describe ~/sorter-sample              # default --sample 400
uv run sorter describe ~/sorter-sample --model qwen3-vl:4b-instruct
```

- Hanya item `needs_model` yang dibaca, dipilih merata per jenis dan per kuartal.
- Yang dikirim ke model hanya cuplikan: halaman pertama PDF (dirender bila hasil scan),
  thumbnail gambar, awal teks, atau daftar nama isi folder.
- Hasil di-cache di `runs/cache.sqlite`; item yang tidak berubah tidak dibaca ulang.
- Output: `runs/<run-id>/descriptors.json` dan `trace.jsonl` (prompt, jawaban, durasi).
- All-local: host selain localhost dan model `-cloud` ditolak, proxy tidak dipakai.


## Jaminan baca-saja

- `scan` dan `describe` hanya membaca daftar isi, metadata, dan (untuk memastikan duplikat) isi file.  Tidak ada yang dipindah, diganti nama, dihapus, atau dibuat di folder yang dipindai.
- `scan` menolak berjalan bila `runs/` berada di dalam folder yang dipindai.
- `--no-hash`: isi file sama sekali tidak dibuka; duplikat hanya dari nama.
- `sample-folder` menolak membuat folder di dalam ~/Downloads, ~/Documents, ~/Desktop, dan
  folder yang sudah berisi.
- Dibuktikan oleh `tests/test_scan.py`: kondisi folder sebelum dan sesudah scan harus sama.

## File

| File | Isi |
|---|---|
| `src/sorter/cli.py` | Perintah `scan`, `describe`, `discover`, `sample-folder`, plus tampilan |
| `src/sorter/config.py` | Semua aturan: penanda project/dataset, pola duplikat, akhiran |
| `src/sorter/schemas.py` | Bentuk data `inventory.json`, `descriptors.json`, `taxonomy.json` |
| `src/sorter/scan.py` | Daftar item level teratas  label triage |
| `src/sorter/duplicates.py` | Duplikat dari pola nama dan isi identik |
| `src/sorter/sample.py` | Folder contoh tiruan Downloads |
| `src/sorter/extract.py` | Cuplikan isi per item untuk model |
| `src/sorter/llm.py` | Klien Ollama, penjaga all-local |
| `src/sorter/describe.py` | Pilih sampel, prompt, panggil model |
| `src/sorter/store.py` | Cache SQLite dan trace JSONL |
| `src/sorter/cluster.py` | Embedding + pengelompokan awal |
| `src/sorter/agent.py` | Loop agen dan tool-nya, kategori tetap dari triage |
| `tests/conftest.py` | Folder contoh dan snapshot, dipakai semua tes |
| `tests/test_scan.py` | Tes baca-saja, duplikat, dan triage |
| `tests/test_describe.py` | Tes describe dengan model palsu (tanpa Ollama) |
| `tests/test_agent.py` | Tes clustering, tool agen, dan loop agen dengan model palsu |
| `tests/test_cli.py` | Tes `describe` dan `discover` dari ujung ke ujung dengan Ollama palsu |

File baru ditambah hanya saat sebuah milestone membutuhkannya.

## Discover (M3): agen menyusun kategori

```bash
ollama pull bge-m3
uv run sorter discover ~/sorter-sample
```

1. **Describe** seperti di atas (item yang sudah dibaca diambil dari cache).
2. **Cluster**: ringkasan diubah jadi embedding lewat `bge-m3` (multibahasa), lalu dikelompokkan
   jadi cluster kecil-kecil (sekitar 4 item per cluster). Cluster dengan 3 jenis item atau lebih
   ditandai campuran. Model lain bisa dicoba dengan `SORTER_EMBED_MODEL=nomic-embed-text`.
3. **Agen**: Qwen memakai tool `inspect_cluster`, `peek_item`, `split`, dan `propose` untuk
   menyusun 3 sampai 12 kategori. Tidak ada `merge`: satu kategori boleh berisi banyak cluster,
   jadi nomor cluster tidak berubah-ubah. Propose yang salah (nomor cluster tidak ada, cluster
   dipakai dua kali atau terlewat) selalu ditolak. Kategori 1 item atau cluster campuran yang
   belum diperiksa hanya diingatkan sekali; kalau agen tetap mengusulkannya, diterima.
   Panggilan gagal yang diulang persis tidak dijalankan lagi, dan setelah 3 giliran gagal
   berturut-turut agen diberi daftar cluster terbaru. Setiap giliran tercatat di `trace.jsonl`.

Model agen bisa dibedakan dari model describe, misalnya model teks yang berpikir dulu:
`uv run sorter discover ~/sorter-sample --agent-model qwen3:8b`.

Item yang jelas dari aturan M1 (app, installer, arsip, project, dataset) masuk kategori tetap tanpa
lewat model. Output: `clusters.json`, `descriptors.json`, dan `taxonomy.json`.

Di terminal terlihat daftar cluster awal yang dilihat agen, status saat agen sedang berpikir, lalu
satu kalimat per langkah (misal "memisahkan 2 item dari cluster 0 → cluster 12 (cv x1,
sertifikat x1)"), lengkap dengan lama tiap giliran dan alasan kalau propose ditolak.