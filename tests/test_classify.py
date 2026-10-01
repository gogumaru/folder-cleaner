"""Tes M5 tanpa model Apple: model palsu mencatat apa yang dikirim kepadanya."""

from conftest import snapshot
from sorter.classify import NONE, evaluate, needs_review
from sorter.config import Settings
from sorter.scan import list_items
from sorter.schemas import Category

CFG = Settings()
CATS = [Category(name="Keuangan", description="Struk, invoice, dan tagihan", tier="important"),
        Category(name="Hiburan", description="Meme dan foto lucu", tier="temporary")]  # fmt: skip


class FakeApple:
    name = "palsu"

    def __init__(self, answers):
        self.answers = answers  # nama kategori per panggilan, atau Exception
        self.calls = []

    def choose(self, instructions, prompt, image, choices):
        self.calls.append({"instructions": instructions, "prompt": prompt,
                           "image": image.read_bytes() if image else None,
                           "choices": choices})  # fmt: skip
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return {"category": answer, "confidence": "tinggi", "reason": "alasan"}


def test_compares_with_agent_and_stays_read_only(sample):
    before = snapshot(sample)
    expected = {"Invoice_Maret.pdf": "Keuangan", "meme.jpg": "Hiburan", "catatan.txt": "Keuangan"}
    model = FakeApple([  # dua jawaban per file; urut nama: catatan, Invoice, meme
        "Keuangan", "Hiburan",   # berubah saat urutan dibalik: ragu
        "Keuangan", "Keuangan",  # konsisten: dipindah otomatis
        NONE, NONE,              # model menolak: ragu
    ])  # fmt: skip

    verdicts = evaluate(list_items(sample, CFG), expected, CATS, model, CFG)

    got = {v.name: (v.expected, v.category, v.review) for v in verdicts}
    assert got == {"Invoice_Maret.pdf": ("Keuangan", "Keuangan", False),
                   "catatan.txt": ("Keuangan", "Keuangan", True),
                   "meme.jpg": ("Hiburan", None, True)}  # fmt: skip
    first, second = model.calls[0], model.calls[1]
    assert first["choices"] == ["Keuangan", "Hiburan", NONE]  # pilihan dibatasi + boleh menolak
    assert second["choices"] == ["Hiburan", "Keuangan", NONE]  # urutan dibalik di panggilan kedua
    assert first["instructions"].index("- Keuangan") < first["instructions"].index("- Hiburan")
    assert second["instructions"].index("- Hiburan") < second["instructions"].index("- Keuangan")
    image_call = next(c for c in model.calls if c["image"])  # meme.jpg dikirim sebagai gambar
    assert "meme" not in image_call["prompt"]  # tanpa nama file, seperti di M2
    assert image_call["image"][:2] == b"\xff\xd8"  # JPEG
    assert snapshot(sample) == before


def test_one_failure_does_not_stop_the_rest(sample):
    expected = {"Invoice_Maret.pdf": "Keuangan", "meme.jpg": "Hiburan", "backup.dat": "Hiburan"}
    model = FakeApple([RuntimeError("GuardrailViolationError"), "Hiburan", "Hiburan"])

    verdicts = {v.name: v for v in evaluate(list_items(sample, CFG), expected, CATS, model, CFG)}

    assert "GuardrailViolationError" in verdicts["Invoice_Maret.pdf"].error
    assert verdicts["Invoice_Maret.pdf"].review  # gagal berarti tidak dipindah otomatis
    assert verdicts["meme.jpg"].category == "Hiburan" and not verdicts["meme.jpg"].review
    assert verdicts["backup.dat"].category is None and verdicts["backup.dat"].review
    assert len(model.calls) == 3  # backup.dat tidak bisa dibaca, jadi tidak dikirim ke model


def test_low_confidence_is_held_even_when_consistent():
    assert not needs_review("Keuangan", "Keuangan", "tinggi")
    assert needs_review("Keuangan", "Keuangan", "rendah")
    assert needs_review("Keuangan", "Hiburan", "tinggi")
    assert needs_review(NONE, NONE, "tinggi")