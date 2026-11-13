"""API Lambda'sı: API Gateway (HTTP API) → Mangum → FastAPI.

Bu Lambda hafiftir (zip paket, torch yok): iş kaydı açar, ön imzalı yükleme
hedefi üretir, durum okur. Model yalnızca worker Lambda'sında yaşar.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from ozetle.service.api import create_app
from ozetle.service.config import Settings
from ozetle.service.jobs import DynamoJobStore
from ozetle.service.logs import configure_logging
from ozetle.service.storage import S3UploadStore


def build_app(settings: Settings | None = None, *, dynamodb: Any = None, s3: Any = None) -> FastAPI:
    import boto3
    from botocore.config import Config

    settings = settings or Settings(backend="aws")
    configure_logging(settings.log_level)
    # Bölgesel uç nokta + sanal barındırma: yeni kovalarda küresel uç nokta
    # 24 saate kadar 307 yönlendirmesi döndürür ve tarayıcıdaki CORS'lu POST
    # yönlendirmeyi izleyemez.
    s3 = s3 or boto3.client(
        "s3", config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"})
    )
    dynamodb = dynamodb or boto3.resource("dynamodb")
    return create_app(
        settings=settings,
        store=DynamoJobStore(settings.table_name, dynamodb),
        uploads=S3UploadStore(settings.upload_bucket, s3, prefix=settings.upload_prefix),
        dispatch=None,  # İşi S3 olayı → SQS başlatır.
    )


_handler: Any = None


def handler(event: dict[str, Any], context: Any) -> Any:
    """Lambda giriş noktası. Uygulama ilk çağrıda kurulur ve konteynerde kalır."""
    global _handler
    if _handler is None:
        from mangum import Mangum

        _handler = Mangum(build_app(), lifespan="off")
    return _handler(event, context)
