"""Servis ayarları — tamamı ortam değişkenlerinden (``OZETLE_`` önekiyle).

Aynı kod iki arka uçla çalışır:

- ``local``: bellek içi iş deposu, diskte yükleme klasörü, arka plan iş
  parçacığında özetleme. ``docker compose up`` veya ``python -m ozetle.service``.
- ``aws``: DynamoDB iş deposu, S3 yükleme (ön imzalı POST), S3 olayı → SQS →
  Lambda worker. Terraform bu değişkenleri Lambda'lara yazar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from ozetle.models.keywords import DEFAULT_EMBEDDING_MODEL
from ozetle.models.summarizer import DEFAULT_MODEL
from ozetle.pdf.extract import MAX_FILE_BYTES, MAX_PAGES


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OZETLE_", extra="ignore")

    backend: Literal["local", "aws"] = "local"

    # Modeller
    model: str = DEFAULT_MODEL
    keyword_model: str = DEFAULT_EMBEDDING_MODEL
    keywords_enabled: bool = True

    # Limitler
    max_upload_bytes: int = MAX_FILE_BYTES
    max_pages: int = MAX_PAGES
    daily_job_limit: int = 300
    """Günlük yeni iş sınırı. Herkese açık demoda maliyet ve kötüye kullanım freni."""
    job_ttl_hours: int = 24
    """İş kayıtları bu süre sonra silinir (DynamoDB TTL)."""
    cache_ttl_days: int = 7
    """Aynı PDF'in (SHA-256) sonucu bu süre boyunca yeniden hesaplanmaz."""
    upload_url_ttl_seconds: int = 600

    # local
    data_dir: Path = Path(".ozetle-data")
    frontend_dir: Path | None = None
    """Verilirse arayüz "/" altında sunulur (yalnızca local)."""

    # aws
    table_name: str = "ozetle-jobs"
    upload_bucket: str = ""
    upload_prefix: str = "uploads/"

    log_level: str = "INFO"
