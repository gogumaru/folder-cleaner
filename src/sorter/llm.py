"""Klien Qwen lewat Ollama yang berjalan di Mac ini.

Aturan all-local ditegakkan di sini: host selain localhost ditolak, model cloud Ollama
(nama berakhiran "-cloud") ditolak, dan proxy sistem tidak pernah dipakai.
"""

import base64
import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

from pydantic import ValidationError

from sorter.schemas import Description

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ModelError(Exception):
    """Jawaban model untuk satu item tidak bisa dipakai. Item lain tetap diproses."""


class ModelUnavailable(Exception):
    """Ollama atau modelnya tidak bisa dipakai sama sekali. Proses dihentikan."""


class Ollama:
    def __init__(self, host: str, model: str, timeout: float) -> None:
        if urlparse(host).hostname not in LOCAL_HOSTS:
            raise ModelUnavailable(f"Host {host} bukan localhost. Isi file tidak boleh keluar Mac.")
        if "cloud" in model.lower():
            raise ModelUnavailable(f"{model} adalah model cloud. Pakai model lokal.")
        self.host = host.rstrip("/")
        self.name = model
        self.timeout = timeout
        # ProxyHandler({}): jangan pernah lewat proxy, walau ada di environment
        self._http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def check(self, model: str | None = None) -> None:
        """Pastikan Ollama jalan dan modelnya sudah di-pull, sebelum mulai."""
        model = model or self.name
        if "cloud" in model.lower():
            raise ModelUnavailable(f"{model} adalah model cloud. Pakai model lokal.")
        self._post("/api/show", {"model": model})

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Ubah teks menjadi vektor (embedding) untuk dikelompokkan."""
        return self._post("/api/embed", {"model": model, "input": texts})["embeddings"]

    def chat(self, messages: list[dict], tools: list[dict], num_ctx: int) -> dict:
        """Satu giliran agen. Balasan bisa berisi tool_calls yang harus dijalankan."""
        body = {
            "model": self.name,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": num_ctx},
        }
        return self._post("/api/chat", body)["message"]

    def describe(self, prompt: str, image: bytes | None = None) -> Description:
        message = {"role": "user", "content": prompt}
        if image:
            message["images"] = [base64.b64encode(image).decode()]
        body = {
            "model": self.name,
            "messages": [message],
            "format": Description.model_json_schema(),  # paksa jawaban sesuai skema
            "stream": False,
            "options": {"temperature": 0},
        }
        content = self._post("/api/chat", body)["message"]["content"]
        try:
            return Description.model_validate_json(content)
        except ValidationError as exc:
            raise ModelError(f"jawaban tidak sesuai skema: {content[:200]}") from exc

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._http.open(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if exc.code == 404:
                model = body.get("model", self.name)
                raise ModelUnavailable(
                    f"Model {model} belum ada. Jalankan: ollama pull {model}"
                ) from exc
            raise ModelError(f"Ollama HTTP {exc.code}: {detail[:200]}") from exc
        except urllib.error.URLError as exc:
            raise ModelUnavailable(
                f"Ollama tidak bisa dihubungi di {self.host}. Buka app Ollama atau jalankan "
                "`ollama serve`."
            ) from exc
        except TimeoutError as exc:
            raise ModelError(f"tidak menjawab dalam {self.timeout:.0f} detik") from exc
        except OSError as exc:  # koneksi putus di tengah jalan, misal Ollama crash
            raise ModelUnavailable(f"Koneksi ke Ollama terputus: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ModelError("balasan Ollama bukan JSON") from exc