"""Yüklenen PDF'lerin geçici deposu.

Tarayıcı dosyayı **doğrudan depoya** yükler; API sunucusu baytları görmez.
AWS'de bu, S3'e ön imzalı POST'tur: Lambda'nın 6 MB istek sınırı aşılır ve
API Lambda'sı büyük dosya taşımak için para/süre harcamaz. Ön imzalı POST
(PUT değil) seçildi çünkü boyut sınırını ``content-length-range`` koşuluyla
S3'ün kendisi uygular.

Her iki uygulama da aynı biçimde hedef döndürür (``url`` + form ``fields``),
böylece arayüz arka ucun hangisi olduğunu bilmez.

PDF'ler işlendikten sonra silinir; S3'te ayrıca 1 günlük yaşam döngüsü kuralı
vardır (worker çökse bile dosya kalmaz).
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ozetle.service.jobs import JOB_ID


@dataclass(frozen=True)
class UploadTarget:
    url: str
    fields: dict[str, str] = field(default_factory=dict)
    max_bytes: int = 0


class UploadStore(Protocol):
    def target(self, job_id: str, *, max_bytes: int, expires_in: int) -> UploadTarget: ...

    def read(self, job_id: str) -> bytes: ...

    def delete(self, job_id: str) -> None: ...


class UploadMissingError(FileNotFoundError):
    pass


class LocalUploadStore:
    """Diskte klasör; yükleme hedefi API'nin kendi uç noktasıdır."""

    def __init__(self, directory: Path, *, base_url: str = "/api/uploads") -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._base_url = base_url

    def _path(self, job_id: str) -> Path:
        if not JOB_ID.match(job_id):
            raise ValueError("geçersiz iş kimliği")
        return self._dir / f"{job_id}.pdf"

    def target(self, job_id: str, *, max_bytes: int, expires_in: int) -> UploadTarget:
        return UploadTarget(url=f"{self._base_url}/{job_id}", max_bytes=max_bytes)

    def save(self, job_id: str, data: bytes) -> None:
        self._path(job_id).write_bytes(data)

    def read(self, job_id: str) -> bytes:
        path = self._path(job_id)
        if not path.exists():
            raise UploadMissingError(job_id)
        return path.read_bytes()

    def delete(self, job_id: str) -> None:
        self._path(job_id).unlink(missing_ok=True)


class S3UploadStore:
    """S3 kovası; ön imzalı POST ile tarayıcıdan doğrudan yükleme."""

    def __init__(self, bucket: str, client: Any, *, prefix: str = "uploads/") -> None:
        self._bucket = bucket
        self._client = client
        self._prefix = prefix

    def key(self, job_id: str) -> str:
        if not JOB_ID.match(job_id):
            raise ValueError("geçersiz iş kimliği")
        return f"{self._prefix}{job_id}.pdf"

    def target(self, job_id: str, *, max_bytes: int, expires_in: int) -> UploadTarget:
        post = self._client.generate_presigned_post(
            Bucket=self._bucket,
            Key=self.key(job_id),
            Fields={"Content-Type": "application/pdf"},
            Conditions=[
                {"Content-Type": "application/pdf"},
                ["content-length-range", 1, max_bytes],
            ],
            ExpiresIn=expires_in,
        )
        return UploadTarget(url=post["url"], fields=dict(post["fields"]), max_bytes=max_bytes)

    def read(self, job_id: str) -> bytes:
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=self.key(job_id))
        except self._client.exceptions.NoSuchKey as exc:
            raise UploadMissingError(job_id) from exc
        data: bytes = obj["Body"].read()
        return data

    def delete(self, job_id: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=self.key(job_id))


def job_id_from_key(key: str, *, prefix: str = "uploads/") -> str | None:
    """S3 olay anahtarından iş kimliği. Olay anahtarları URL kodludur."""
    key = urllib.parse.unquote_plus(key)
    if not key.startswith(prefix) or not key.endswith(".pdf"):
        return None
    job_id = key[len(prefix) : -len(".pdf")]
    return job_id if JOB_ID.match(job_id) else None
