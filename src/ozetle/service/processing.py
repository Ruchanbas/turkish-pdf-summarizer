"""Worker çekirdeği: bir işi baştan sona işler.

Hem local arka uçta (arka plan iş parçacığı) hem AWS'de (SQS tetiklemeli
Lambda) aynı fonksiyon çalışır.

SQS mesajı **en az bir kez** teslim eder; aynı iş iki kez gelebilir. Bu
yüzden işleme idempotenttir: bitmiş iş tekrar işlenmez.

Hata sınıfları farklı ele alınır:

- PDF hataları (parola, taranmış belge, bozuk dosya): kullanıcının düzeltmesi
  gereken durumlar. İş ``failed`` olur, mesaj kullanıcıya gösterilir,
  yeniden denenmez.
- Model yükleme hatası: geçici olabilir (disk, bellek). İstisna yükselir,
  SQS mesajı yeniden dener; iki denemeden sonra DLQ'ya düşer ve alarm çalar.
- Diğer beklenmeyen hatalar: iş ``failed`` olur, ayrıntı loglanır,
  kullanıcıya genel bir mesaj gösterilir (iç ayrıntı sızdırılmaz).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ozetle.models.keywords import KeywordExtractor
from ozetle.models.loader import ModelLoadError
from ozetle.models.summarizer import Summarizer, summarize_document
from ozetle.pdf.extract import PdfError
from ozetle.pipeline import prepare
from ozetle.service.jobs import Job, JobStatus, JobStore
from ozetle.service.storage import UploadMissingError, UploadStore

logger = logging.getLogger(__name__)

#: Sonuçta parça başına saklanan kaynak metin önizlemesi. DynamoDB öğesi en
#: fazla 400 KB olabilir; 60 sayfalık belgenin tam metni bu sınırı zorlar.
SECTION_PREVIEW_CHARS = 600

GENERIC_ERROR = "Belge işlenirken beklenmeyen bir hata oluştu. Lütfen tekrar deneyin."


@dataclass
class Engine:
    """Modeller. Tembel kurulur: ilk iş gelene kadar hiçbir şey yüklenmez."""

    summarizer: Summarizer
    keywords: KeywordExtractor | None = None


def _preview(text: str, limit: int = SECTION_PREVIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit] + " …"


def build_result(
    pdf_bytes: bytes,
    engine: Engine,
    *,
    max_pages: int,
    keyword_count: int = 8,
) -> dict[str, Any]:
    """PDF baytlarından API'nin döndürdüğü sonuç sözlüğünü üretir.

    Raises:
        PdfError: PDF okunamadı veya içinden metin çıkmadı.
    """
    summarizer = engine.summarizer
    timings: dict[str, float] = {}

    started = time.perf_counter()
    doc = prepare(
        pdf_bytes,
        max_tokens=summarizer.max_input_tokens - 16,
        max_pages=max_pages,
        token_counter=summarizer.count_tokens,
    )
    timings["prepare"] = time.perf_counter() - started
    if doc.is_empty:
        raise PdfError("Temizlik sonrası özetlenecek metin kalmadı.")

    summary = summarize_document([c.text for c in doc.chunks], summarizer)
    timings["summarize"] = summary.seconds

    keywords: list[dict[str, Any]] = []
    if engine.keywords is not None:
        started = time.perf_counter()
        keywords = [
            {"phrase": k.phrase, "score": k.score}
            for k in engine.keywords.extract(doc.text, top_n=keyword_count)
        ]
        timings["keywords"] = time.perf_counter() - started

    warnings = [*doc.warnings, *summary.warnings]
    return {
        "summary": summary.summary,
        "keywords": keywords,
        "sections": [
            {"index": c.index, "source": _preview(c.text), "summary": partial}
            for c, partial in zip(doc.chunks, summary.partials, strict=True)
        ],
        "stats": {
            "model": summary.model,
            "pages": doc.total_pages,
            "pages_read": doc.pages_read,
            "chunks": len(doc.chunks),
            "rounds": summary.rounds,
            "characters": doc.metrics.as_dict(),
            "seconds": {k: round(v, 2) for k, v in timings.items()},
        },
        "warnings": warnings,
    }


def process_job(
    job_id: str,
    *,
    store: JobStore,
    uploads: UploadStore,
    engine_factory: Callable[[], Engine],
    max_pages: int,
    cache_ttl_seconds: int,
) -> Job | None:
    """Bir işi işler ve son halini döndürür. İş yoksa ``None``.

    Raises:
        ModelLoadError: Yeniden denenebilir hata; çağıran (SQS) tekrar dener.
    """
    job = store.get(job_id)
    if job is None:
        logger.warning("İş bulunamadı (süresi dolmuş olabilir)", extra={"job_id": job_id})
        return None
    if job.status.terminal:
        logger.info("İş zaten bitmiş, atlanıyor", extra={"job_id": job_id})
        return job

    job = store.update(job_id, status=JobStatus.PROCESSING)
    try:
        data = uploads.read(job_id)
    except UploadMissingError:
        return store.update(job_id, status=JobStatus.FAILED, error="Yüklenen dosya bulunamadı.")

    sha256 = hashlib.sha256(data).hexdigest()
    cached = store.get_cached(sha256)
    if cached is not None:
        uploads.delete(job_id)
        logger.info("Önbellekten döndü", extra={"job_id": job_id, "sha256": sha256})
        return store.update(
            job_id, status=JobStatus.DONE, sha256=sha256, result=cached, cached=True
        )

    started = time.perf_counter()
    try:
        result = build_result(data, engine_factory(), max_pages=max_pages)
    except ModelLoadError:
        # Dosya silinmez: mesaj yeniden denendiğinde gerekecek.
        store.update(job_id, status=JobStatus.QUEUED)
        raise
    except PdfError as exc:
        uploads.delete(job_id)
        logger.info("PDF reddedildi: %s", exc, extra={"job_id": job_id})
        return store.update(job_id, status=JobStatus.FAILED, sha256=sha256, error=str(exc))
    except Exception:
        uploads.delete(job_id)
        logger.exception("İş başarısız", extra={"job_id": job_id})
        return store.update(job_id, status=JobStatus.FAILED, sha256=sha256, error=GENERIC_ERROR)

    uploads.delete(job_id)
    store.put_cached(sha256, result, cache_ttl_seconds)
    logger.info(
        "İş tamamlandı",
        extra={
            "job_id": job_id,
            "seconds": round(time.perf_counter() - started, 2),
            "pages": result["stats"]["pages_read"],
            "chunks": result["stats"]["chunks"],
            "model": result["stats"]["model"],
        },
    )
    return store.update(job_id, status=JobStatus.DONE, sha256=sha256, result=result)
