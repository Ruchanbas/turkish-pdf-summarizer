"""İş kayıtları, sonuç önbelleği ve günlük kota.

Tek bir DynamoDB tablosu üç kayıt türü taşır (``pk`` ön ekiyle ayrılır):

    job#<id>         iş durumu ve sonucu
    cache#<sha256>   aynı PDF'in sonucu — tekrar yüklemede model çalışmaz
    quota#<gün>      o günkü iş sayacı (atomik artırım)

Hepsinin ``expires_at`` alanı vardır; DynamoDB TTL süresi dolanları siler.
Sonuç, DynamoDB'nin float yerine Decimal isteme zorunluluğuna takılmamak için
JSON metin olarak saklanır.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class JobStatus(StrEnum):
    AWAITING_UPLOAD = "awaiting_upload"
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED)


class JobNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class Job:
    id: str
    status: JobStatus
    filename: str
    size: int
    created_at: float
    updated_at: float
    expires_at: int
    sha256: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    cached: bool = False

    @classmethod
    def new(cls, *, filename: str, size: int, ttl_seconds: int) -> Job:
        now = time.time()
        return cls(
            id=uuid.uuid4().hex,
            status=JobStatus.AWAITING_UPLOAD,
            filename=filename,
            size=size,
            created_at=now,
            updated_at=now,
            expires_at=int(now) + ttl_seconds,
        )

    def public(self) -> dict[str, Any]:
        """API'de döndürülen görünüm."""
        data = asdict(self)
        data["status"] = self.status.value
        data.pop("expires_at")
        return data

    def to_item(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "pk": f"job#{self.id}",
            "status": self.status.value,
            "filename": self.filename,
            "size": self.size,
            "created_at": repr(self.created_at),
            "updated_at": repr(self.updated_at),
            "expires_at": self.expires_at,
            "cached": self.cached,
        }
        if self.sha256:
            item["sha256"] = self.sha256
        if self.result is not None:
            item["result_json"] = json.dumps(self.result, ensure_ascii=False)
        if self.error:
            item["error"] = self.error
        return item

    @classmethod
    def from_item(cls, item: dict[str, Any]) -> Job:
        return cls(
            id=str(item["pk"]).removeprefix("job#"),
            status=JobStatus(item["status"]),
            filename=str(item["filename"]),
            size=int(item["size"]),
            created_at=float(item["created_at"]),
            updated_at=float(item["updated_at"]),
            expires_at=int(item["expires_at"]),
            sha256=item.get("sha256"),
            result=json.loads(item["result_json"]) if "result_json" in item else None,
            error=item.get("error"),
            cached=bool(item.get("cached", False)),
        )


class JobStore(Protocol):
    def create(self, job: Job) -> None: ...

    def get(self, job_id: str) -> Job | None: ...

    def update(self, job_id: str, **changes: Any) -> Job: ...

    def get_cached(self, sha256: str) -> dict[str, Any] | None: ...

    def put_cached(self, sha256: str, result: dict[str, Any], ttl_seconds: int) -> None: ...

    def increment_daily(self, day: str, ttl_seconds: int) -> int: ...


# ---------------------------------------------------------------------------
# Bellek içi (local arka uç ve testler)
# ---------------------------------------------------------------------------


@dataclass
class InMemoryJobStore:
    _jobs: dict[str, Job] = field(default_factory=dict)
    _cache: dict[str, tuple[dict[str, Any], float]] = field(default_factory=dict)
    _quota: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            job = replace(self._jobs[job_id], updated_at=time.time(), **changes)
            self._jobs[job_id] = job
            return job

    def get_cached(self, sha256: str) -> dict[str, Any] | None:
        with self._lock:
            hit = self._cache.get(sha256)
            if hit is None or hit[1] < time.time():
                return None
            return hit[0]

    def put_cached(self, sha256: str, result: dict[str, Any], ttl_seconds: int) -> None:
        with self._lock:
            self._cache[sha256] = (result, time.time() + ttl_seconds)

    def increment_daily(self, day: str, ttl_seconds: int) -> int:
        with self._lock:
            self._quota[day] = self._quota.get(day, 0) + 1
            return self._quota[day]


# ---------------------------------------------------------------------------
# DynamoDB
# ---------------------------------------------------------------------------


class DynamoJobStore:
    """DynamoDB üzerinde iş deposu. ``client`` bir boto3 DynamoDB istemcisidir."""

    def __init__(self, table_name: str, client: Any) -> None:
        self._table = client.Table(table_name) if hasattr(client, "Table") else client

    def create(self, job: Job) -> None:
        self._table.put_item(Item=job.to_item(), ConditionExpression="attribute_not_exists(pk)")

    def get(self, job_id: str) -> Job | None:
        item = self._table.get_item(Key={"pk": f"job#{job_id}"}, ConsistentRead=True).get("Item")
        return Job.from_item(item) if item else None

    def update(self, job_id: str, **changes: Any) -> Job:
        current = self.get(job_id)
        if current is None:
            raise JobNotFoundError(job_id)
        job = replace(current, updated_at=time.time(), **changes)
        self._table.put_item(Item=job.to_item())
        return job

    def get_cached(self, sha256: str) -> dict[str, Any] | None:
        item = self._table.get_item(Key={"pk": f"cache#{sha256}"}).get("Item")
        # TTL silmesi 48 saate kadar gecikebilir; süresi geçen kayıt okunmaz.
        if not item or int(item["expires_at"]) < time.time():
            return None
        result: dict[str, Any] = json.loads(item["result_json"])
        return result

    def put_cached(self, sha256: str, result: dict[str, Any], ttl_seconds: int) -> None:
        self._table.put_item(
            Item={
                "pk": f"cache#{sha256}",
                "result_json": json.dumps(result, ensure_ascii=False),
                "expires_at": int(time.time()) + ttl_seconds,
            }
        )

    def increment_daily(self, day: str, ttl_seconds: int) -> int:
        response = self._table.update_item(
            Key={"pk": f"quota#{day}"},
            UpdateExpression="ADD #n :one SET expires_at = if_not_exists(expires_at, :exp)",
            ExpressionAttributeNames={"#n": "count"},
            ExpressionAttributeValues={":one": 1, ":exp": int(time.time()) + ttl_seconds},
            ReturnValues="UPDATED_NEW",
        )
        return int(response["Attributes"]["count"])
