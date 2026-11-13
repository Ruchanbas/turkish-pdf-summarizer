"""Worker Lambda'sı: S3 olayı → SQS → bu fonksiyon → model.

Konteyner imajı olarak dağıtılır (torch + modeller imajın içinde, çalışma
anında indirme yok). Modeller ilk iş geldiğinde yüklenir ve konteyner sıcak
kaldıkça bellekte durur.

Kısmi toplu hata bildirimi (``ReportBatchItemFailures``) kullanılır: bir
mesaj başarısız olursa yalnızca o mesaj yeniden denenir.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ozetle.service.config import Settings
from ozetle.service.jobs import DynamoJobStore, JobStore
from ozetle.service.logs import configure_logging
from ozetle.service.processing import Engine, process_job
from ozetle.service.storage import S3UploadStore, UploadStore, job_id_from_key

logger = logging.getLogger(__name__)


@dataclass
class WorkerDeps:
    settings: Settings
    store: JobStore
    uploads: UploadStore
    engine_factory: Callable[[], Engine]


def build_deps(settings: Settings | None = None) -> WorkerDeps:
    import boto3

    from ozetle.service.engine import cached_engine_factory

    settings = settings or Settings(backend="aws")
    configure_logging(settings.log_level)
    return WorkerDeps(
        settings=settings,
        store=DynamoJobStore(settings.table_name, boto3.resource("dynamodb")),
        uploads=S3UploadStore(
            settings.upload_bucket, boto3.client("s3"), prefix=settings.upload_prefix
        ),
        engine_factory=cached_engine_factory(settings),
    )


def job_ids_from_record(record: dict[str, Any], *, prefix: str) -> list[str]:
    """SQS kaydındaki S3 olay bildiriminden iş kimlikleri.

    S3, bildirim hedefi kurulurken bir ``s3:TestEvent`` gönderir; o atlanır.
    """
    body = json.loads(record["body"])
    if body.get("Event") == "s3:TestEvent":
        return []
    ids = []
    for s3_record in body.get("Records", []):
        job_id = job_id_from_key(s3_record["s3"]["object"]["key"], prefix=prefix)
        if job_id:
            ids.append(job_id)
    return ids


_deps: WorkerDeps | None = None


def handle(event: dict[str, Any], deps: WorkerDeps) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    for record in event.get("Records", []):
        try:
            for job_id in job_ids_from_record(record, prefix=deps.settings.upload_prefix):
                process_job(
                    job_id,
                    store=deps.store,
                    uploads=deps.uploads,
                    engine_factory=deps.engine_factory,
                    max_pages=deps.settings.max_pages,
                    cache_ttl_seconds=deps.settings.cache_ttl_days * 86400,
                )
        except Exception:
            logger.exception(
                "Mesaj işlenemedi, yeniden denenecek", extra={"message_id": record.get("messageId")}
            )
            failures.append({"itemIdentifier": record["messageId"]})
    return {"batchItemFailures": failures}


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda giriş noktası."""
    global _deps
    if _deps is None:
        _deps = build_deps()
    return handle(event, _deps)
