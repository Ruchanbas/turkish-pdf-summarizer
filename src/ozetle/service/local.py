"""Local arka uç: tek süreçte API + arka plan worker.

Bulut gerektirmeden tüm akışı çalıştırır (``docker compose up``). İş deposu
bellek içidir; sunucu yeniden başlarsa işler kaybolur — bu bir geliştirme ve
demo ortamıdır.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI

from ozetle.models.loader import ModelLoadError
from ozetle.service.api import create_app
from ozetle.service.config import Settings
from ozetle.service.engine import cached_engine_factory
from ozetle.service.jobs import InMemoryJobStore, JobStatus, JobStore
from ozetle.service.processing import Engine, process_job
from ozetle.service.storage import LocalUploadStore

logger = logging.getLogger(__name__)


def build_local_app(
    settings: Settings,
    *,
    engine_factory: Callable[[], Engine] | None = None,
    store: JobStore | None = None,
    synchronous: bool = False,
) -> FastAPI:
    """Local uygulamayı kurar.

    Args:
        engine_factory: Testlerde sahte model vermek için.
        synchronous: İşi istek içinde işler (testler için); varsayılan olarak
            tek iş parçacıklı bir havuzda sırayla işlenir — model bellekte bir
            kez durur ve iki iş aynı anda CPU için yarışmaz.
    """
    store = store or InMemoryJobStore()
    uploads = LocalUploadStore(settings.data_dir / "uploads")
    factory = engine_factory or cached_engine_factory(settings)

    def run(job_id: str) -> None:
        try:
            process_job(
                job_id,
                store=store,
                uploads=uploads,
                engine_factory=factory,
                max_pages=settings.max_pages,
                cache_ttl_seconds=settings.cache_ttl_days * 86400,
            )
        except ModelLoadError as exc:
            # Local'de yeniden deneyen bir kuyruk yok; kullanıcıya söyle.
            logger.exception("Model yüklenemedi")
            store.update(job_id, status=JobStatus.FAILED, error=f"Model yüklenemedi: {exc}")

    if synchronous:
        dispatch = run
    else:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ozetle-worker")

        def dispatch(job_id: str) -> None:
            executor.submit(run, job_id)

    return create_app(settings=settings, store=store, uploads=uploads, dispatch=dispatch)
