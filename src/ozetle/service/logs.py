"""JSON yapılandırılmış loglama.

Her satır tek bir JSON nesnesidir; CloudWatch Logs Insights alanları doğrudan
sorgulayabilir::

    fields @timestamp, job_id, seconds, chunks
    | filter msg = "İş tamamlandı"
    | stats avg(seconds), pct(seconds, 95) by bin(1h)

``logger.info("...", extra={"job_id": ...})`` ile verilen alanlar nesneye eklenir.
"""

from __future__ import annotations

import json
import logging
from typing import Any

_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Kök logger'ı JSON'a çevirir. Tekrar çağrılması güvenlidir.

    Lambda çalışma ortamı kök logger'a kendi handler'ını ekler; yenisini
    eklemek yerine mevcut handler'ların biçimi değiştirilir (aksi halde her
    satır iki kez yazılır).
    """
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    for handler in root.handlers:
        handler.setFormatter(JsonFormatter())
