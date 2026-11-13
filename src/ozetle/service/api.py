"""HTTP API (FastAPI). Local'de uvicorn, AWS'de Lambda (Mangum) üzerinde çalışır.

Akış (arayüz her iki arka uçta da aynı adımları izler)::

    POST /api/jobs            {filename, size, sha256?}
        → 201 {job, upload: {url, fields}}     (yükleme hedefi)
        → 201 {job: done, upload: null}        (aynı PDF önbellekte)
    <tarayıcı PDF'i upload.url'e POST eder>    (AWS: doğrudan S3)
    GET  /api/jobs/{id}                        (durum + sonuç; arayüz yoklar)

Uzun iş isteği bloklamaz: API iş kaydını açıp hemen döner, özetleme
worker'da yapılır. API Gateway'in 29 saniyelik istek sınırı bu yüzden sorun
olmaz.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import replace
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ozetle import __version__
from ozetle.service.config import Settings
from ozetle.service.jobs import JOB_ID, Job, JobStatus, JobStore
from ozetle.service.storage import LocalUploadStore, UploadStore


class CreateJobRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    size: int = Field(gt=0, description="Bayt cinsinden dosya boyutu")
    sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Dosyanın SHA-256'sı. Verilirse aynı PDF yeniden işlenmez.",
    )


def _today() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")


def create_app(
    *,
    settings: Settings,
    store: JobStore,
    uploads: UploadStore,
    dispatch: Callable[[str], None] | None = None,
) -> FastAPI:
    """Uygulamayı kurar.

    Args:
        dispatch: Local arka uçta yükleme bitince işi başlatan fonksiyon.
            AWS'de ``None``: iş, S3 olayının SQS'e düşmesiyle başlar.
    """
    app = FastAPI(
        title="ozetle",
        version=__version__,
        description="Türkçe PDF özetleme servisi",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    max_mb = settings.max_upload_bytes // (1024 * 1024)

    # Yönlendirici kaynaklı hatalar (eşleşmeyen yol, yanlış yöntem) da aynı
    # {"error": ...} biçiminde ve Türkçe dönsün.
    default_messages = {404: "İstenen kaynak bulunamadı.", 405: "Bu yöntem desteklenmiyor."}

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        if not isinstance(exc, HTTPException) or detail in ("Not Found", "Method Not Allowed"):
            detail = default_messages.get(exc.status_code, detail)
        return JSONResponse({"error": detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [".".join(str(p) for p in e["loc"][1:]) for e in exc.errors()]
        return JSONResponse({"error": "Geçersiz istek.", "fields": fields}, status_code=422)

    def _job_or_404(job_id: str) -> Job:
        job = store.get(job_id) if JOB_ID.match(job_id) else None
        if job is None:
            raise HTTPException(404, "İş bulunamadı. Süresi dolmuş olabilir.")
        return job

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "backend": settings.backend,
            "model": settings.model,
            "limits": {"max_upload_mb": max_mb, "max_pages": settings.max_pages},
        }

    @app.post("/api/jobs", status_code=201)
    def create_job(body: CreateJobRequest) -> dict[str, Any]:
        if body.size > settings.max_upload_bytes:
            raise HTTPException(413, f"Dosya {max_mb} MB sınırını aşıyor.")
        if not body.filename.lower().endswith(".pdf"):
            raise HTTPException(415, "Yalnızca PDF dosyaları kabul edilir.")
        if store.increment_daily(_today(), 2 * 86400) > settings.daily_job_limit:
            raise HTTPException(429, "Günlük işlem sınırına ulaşıldı. Yarın tekrar deneyin.")

        job = Job.new(
            filename=body.filename, size=body.size, ttl_seconds=settings.job_ttl_hours * 3600
        )
        if body.sha256 and (cached := store.get_cached(body.sha256)) is not None:
            job = replace(
                job, status=JobStatus.DONE, sha256=body.sha256, result=cached, cached=True
            )
            store.create(job)
            return {"job": job.public(), "upload": None}

        store.create(job)
        target = uploads.target(
            job.id,
            max_bytes=settings.max_upload_bytes,
            expires_in=settings.upload_url_ttl_seconds,
        )
        return {
            "job": job.public(),
            "upload": {"url": target.url, "fields": target.fields, "max_bytes": target.max_bytes},
        }

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return _job_or_404(job_id).public()

    if isinstance(uploads, LocalUploadStore):
        local_uploads = uploads

        @app.post("/api/uploads/{job_id}", status_code=202)
        async def upload(job_id: str, file: Annotated[UploadFile, File()]) -> dict[str, Any]:
            job = _job_or_404(job_id)
            if job.status != JobStatus.AWAITING_UPLOAD:
                raise HTTPException(409, "Bu iş için dosya zaten yüklendi.")
            data = await file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, f"Dosya {max_mb} MB sınırını aşıyor.")
            if b"%PDF-" not in data[:1024]:
                raise HTTPException(415, "Dosya bir PDF değil.")
            local_uploads.save(job_id, data)
            job = store.update(job_id, status=JobStatus.QUEUED, size=len(data))
            if dispatch is not None:
                dispatch(job_id)
            return {"job": job.public()}

    if settings.frontend_dir is not None:
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")

    return app
