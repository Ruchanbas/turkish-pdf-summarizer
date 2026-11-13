"""Servis katmanı testleri: iş deposu, yükleme deposu, worker, API, Lambda.

AWS servisleri moto ile taklit edilir; hiçbir test gerçek hesaba gitmez.
İş deposu testleri hem bellek içi hem DynamoDB uygulamasında aynen koşar —
local'de geçen davranış AWS'de de geçmeli.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from conftest import FakeKeywords, FakeSummarizer
from ozetle.models.loader import ModelLoadError
from ozetle.service import lambda_api, lambda_worker
from ozetle.service.config import Settings
from ozetle.service.jobs import (
    DynamoJobStore,
    InMemoryJobStore,
    Job,
    JobNotFoundError,
    JobStatus,
    JobStore,
)
from ozetle.service.local import build_local_app
from ozetle.service.logs import JsonFormatter, configure_logging
from ozetle.service.processing import GENERIC_ERROR, Engine, build_result, process_job
from ozetle.service.storage import (
    LocalUploadStore,
    S3UploadStore,
    UploadMissingError,
    job_id_from_key,
)

TABLE = "ozetle-jobs-test"
BUCKET = "ozetle-uploads-test"


@pytest.fixture
def aws(aws_env: None) -> Iterator[dict[str, Any]]:
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="eu-central-1")
        dynamodb.create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        s3 = boto3.client("s3", region_name="eu-central-1")
        s3.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"}
        )
        yield {"dynamodb": dynamodb, "s3": s3}


@pytest.fixture(params=["memory", "dynamo"])
def store(request: pytest.FixtureRequest) -> JobStore:
    if request.param == "memory":
        return InMemoryJobStore()
    aws = request.getfixturevalue("aws")
    return DynamoJobStore(TABLE, aws["dynamodb"])


def _job(**kw: Any) -> Job:
    return replace(Job.new(filename="makale.pdf", size=1234, ttl_seconds=3600), **kw)


# ---------------------------------------------------------------------------
# İş deposu (her iki uygulamada)
# ---------------------------------------------------------------------------


class TestJobStore:
    def test_olustur_ve_oku(self, store: JobStore) -> None:
        job = _job()
        store.create(job)
        assert store.get(job.id) == job
        assert store.get("0" * 32) is None

    def test_guncelle(self, store: JobStore) -> None:
        job = _job()
        store.create(job)
        result = {"summary": "Özet.", "stats": {"seconds": {"total": 1.25}}}
        updated = store.update(job.id, status=JobStatus.DONE, result=result, sha256="a" * 64)
        assert updated.updated_at >= job.updated_at
        again = store.get(job.id)
        assert again is not None
        assert again.status is JobStatus.DONE
        assert again.result == result  # float'lar Decimal'e takılmadan döner
        assert again.sha256 == "a" * 64

    def test_olmayan_is_guncellenemez(self, store: JobStore) -> None:
        with pytest.raises(JobNotFoundError):
            store.update("0" * 32, status=JobStatus.DONE)

    def test_onbellek(self, store: JobStore) -> None:
        assert store.get_cached("b" * 64) is None
        store.put_cached("b" * 64, {"summary": "Ç"}, ttl_seconds=60)
        assert store.get_cached("b" * 64) == {"summary": "Ç"}

    def test_suresi_dolan_onbellek_okunmaz(self, store: JobStore) -> None:
        store.put_cached("c" * 64, {"summary": "x"}, ttl_seconds=-10)
        assert store.get_cached("c" * 64) is None

    def test_gunluk_sayac_atomik_artar(self, store: JobStore) -> None:
        assert [store.increment_daily("2026-11-13", 60) for _ in range(3)] == [1, 2, 3]
        assert store.increment_daily("2026-11-14", 60) == 1


class TestJob:
    def test_public_gorunum(self) -> None:
        data = _job().public()
        assert data["status"] == "awaiting_upload"
        assert "expires_at" not in data

    def test_terminal_durumlar(self) -> None:
        assert JobStatus.DONE.terminal and JobStatus.FAILED.terminal
        assert not JobStatus.PROCESSING.terminal


# ---------------------------------------------------------------------------
# Yükleme deposu
# ---------------------------------------------------------------------------


class TestUploadStore:
    def test_local(self, tmp_path) -> None:
        uploads = LocalUploadStore(tmp_path)
        job_id = "a" * 32
        assert uploads.target(job_id, max_bytes=10, expires_in=60).url == f"/api/uploads/{job_id}"
        uploads.save(job_id, b"%PDF-1.4")
        assert uploads.read(job_id) == b"%PDF-1.4"
        uploads.delete(job_id)
        with pytest.raises(UploadMissingError):
            uploads.read(job_id)
        uploads.delete(job_id)  # ikinci silme sorun değil

    def test_local_yol_gecisi_engellenir(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            LocalUploadStore(tmp_path).save("../../etc/passwd", b"x")

    def test_s3_on_imzali_post(self, aws) -> None:
        uploads = S3UploadStore(BUCKET, aws["s3"])
        target = uploads.target("d" * 32, max_bytes=1000, expires_in=60)
        assert BUCKET in target.url
        assert target.fields["key"] == f"uploads/{'d' * 32}.pdf"
        assert target.fields["Content-Type"] == "application/pdf"
        policy = json.loads(__import__("base64").b64decode(target.fields["policy"]).decode())
        assert ["content-length-range", 1, 1000] in policy["conditions"]

    def test_s3_oku_sil(self, aws) -> None:
        uploads = S3UploadStore(BUCKET, aws["s3"])
        job_id = "e" * 32
        aws["s3"].put_object(Bucket=BUCKET, Key=uploads.key(job_id), Body=b"%PDF-x")
        assert uploads.read(job_id) == b"%PDF-x"
        uploads.delete(job_id)
        with pytest.raises(UploadMissingError):
            uploads.read(job_id)

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            (f"uploads/{'a' * 32}.pdf", "a" * 32),
            (f"uploads%2F{'a' * 32}.pdf", "a" * 32),  # olay anahtarları URL kodlu
            (f"diger/{'a' * 32}.pdf", None),
            ("uploads/kotu.pdf", None),
            (f"uploads/{'a' * 32}.txt", None),
        ],
    )
    def test_anahtardan_is_kimligi(self, key: str, expected: str | None) -> None:
        assert job_id_from_key(key) == expected


# ---------------------------------------------------------------------------
# Worker çekirdeği
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> Engine:
    return Engine(summarizer=FakeSummarizer(), keywords=FakeKeywords())


def _process(store: JobStore, uploads: Any, engine: Engine, job_id: str) -> Job | None:
    return process_job(
        job_id,
        store=store,
        uploads=uploads,
        engine_factory=lambda: engine,
        max_pages=60,
        cache_ttl_seconds=3600,
    )


@pytest.fixture
def local_uploads(tmp_path) -> LocalUploadStore:
    return LocalUploadStore(tmp_path / "uploads")


class TestProcessing:
    def test_basarili_is(self, store, local_uploads, engine, article_bytes) -> None:
        job = _job()
        store.create(job)
        local_uploads.save(job.id, article_bytes)
        done = _process(store, local_uploads, engine, job.id)
        assert done is not None and done.status is JobStatus.DONE
        assert done.sha256 == hashlib.sha256(article_bytes).hexdigest()
        result = done.result or {}
        assert result["summary"]
        assert [k["phrase"] for k in result["keywords"]] == ["fizyoterapi", "fonksiyonel skor"]
        assert len(result["sections"]) == result["stats"]["chunks"] > 1
        assert all(len(s["source"]) <= 602 for s in result["sections"])
        assert result["stats"]["model"] == "fake/summarizer"
        assert set(result["stats"]["seconds"]) == {"prepare", "summarize", "keywords"}
        with pytest.raises(UploadMissingError):  # PDF işlendikten sonra silinir
            local_uploads.read(job.id)

    def test_ayni_pdf_onbellekten(self, store, local_uploads, engine, article_bytes) -> None:
        first, second = _job(), _job()
        for job in (first, second):
            store.create(job)
            local_uploads.save(job.id, article_bytes)
        _process(store, local_uploads, engine, first.id)
        calls = len(engine.summarizer.calls)  # type: ignore[attr-defined]
        done = _process(store, local_uploads, engine, second.id)
        assert done is not None and done.cached
        assert len(engine.summarizer.calls) == calls  # type: ignore[attr-defined]

    def test_pdf_hatasi_kullaniciya_gosterilir(self, store, local_uploads, engine) -> None:
        job = _job()
        store.create(job)
        local_uploads.save(job.id, b"bu bir pdf degil")
        failed = _process(store, local_uploads, engine, job.id)
        assert failed is not None and failed.status is JobStatus.FAILED
        assert failed.error == "Dosya bir PDF değil."

    def test_bitmis_is_tekrar_islenmez(self, store, local_uploads, engine) -> None:
        """SQS en az bir kez teslim eder; aynı mesaj iki kez gelebilir."""
        job = _job(status=JobStatus.DONE, result={"summary": "eski"})
        store.create(job)
        assert _process(store, local_uploads, engine, job.id) == job
        assert engine.summarizer.calls == []  # type: ignore[attr-defined]

    def test_olmayan_is(self, store, local_uploads, engine) -> None:
        assert _process(store, local_uploads, engine, "f" * 32) is None

    def test_dosya_yoksa_basarisiz(self, store, local_uploads, engine) -> None:
        job = _job()
        store.create(job)
        failed = _process(store, local_uploads, engine, job.id)
        assert failed is not None and failed.error == "Yüklenen dosya bulunamadı."

    def test_model_hatasi_yeniden_denenir(self, store, local_uploads, article_bytes) -> None:
        job = _job()
        store.create(job)
        local_uploads.save(job.id, article_bytes)

        def bozuk() -> Engine:
            raise ModelLoadError("disk dolu")

        with pytest.raises(ModelLoadError):
            process_job(
                job.id,
                store=store,
                uploads=local_uploads,
                engine_factory=bozuk,
                max_pages=60,
                cache_ttl_seconds=60,
            )
        assert store.get(job.id).status is JobStatus.QUEUED  # type: ignore[union-attr]
        assert local_uploads.read(job.id) == article_bytes  # yeniden deneme için durur

    def test_beklenmeyen_hata_ic_ayrinti_sizdirmaz(
        self, store, local_uploads, article_bytes, caplog
    ) -> None:
        class Patlayan(FakeSummarizer):
            def summarize_batch(self, texts):
                raise RuntimeError("CUDA out of memory at 0xdeadbeef")

        job = _job()
        store.create(job)
        local_uploads.save(job.id, article_bytes)
        failed = _process(store, local_uploads, Engine(summarizer=Patlayan()), job.id)
        assert failed is not None and failed.error == GENERIC_ERROR
        assert "0xdeadbeef" in caplog.text  # ayrıntı logda, kullanıcıda değil

    def test_anahtar_kelime_kapatilabilir(self, article_bytes) -> None:
        result = build_result(article_bytes, Engine(summarizer=FakeSummarizer()), max_pages=60)
        assert result["keywords"] == []
        assert "keywords" not in result["stats"]["seconds"]


# ---------------------------------------------------------------------------
# HTTP API (local arka uç)
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(backend="local", data_dir=tmp_path, daily_job_limit=50)


@pytest.fixture
def client(settings: Settings, engine: Engine) -> TestClient:
    app = build_local_app(settings, engine_factory=lambda: engine, synchronous=True)
    return TestClient(app)


def _create(client: TestClient, **body: Any) -> Any:
    payload = {"filename": "makale.pdf", "size": 1000, **body}
    return client.post("/api/jobs", json=payload)


class TestApi:
    def test_saglik(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["limits"]["max_upload_mb"] == 20

    def test_tam_akis(self, client: TestClient, article_bytes: bytes) -> None:
        created = _create(client, size=len(article_bytes))
        assert created.status_code == 201
        job_id = created.json()["job"]["id"]
        upload = created.json()["upload"]
        assert upload["url"] == f"/api/uploads/{job_id}"

        uploaded = client.post(
            upload["url"], files={"file": ("makale.pdf", article_bytes, "application/pdf")}
        )
        assert uploaded.status_code == 202

        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "done"
        assert job["result"]["summary"]
        assert job["result"]["sections"]

    def test_onbellek_ile_yuklemesiz_sonuc(self, client: TestClient, article_bytes: bytes) -> None:
        sha = hashlib.sha256(article_bytes).hexdigest()
        first = _create(client, sha256=sha).json()
        client.post(
            first["upload"]["url"], files={"file": ("m.pdf", article_bytes, "application/pdf")}
        )
        second = _create(client, sha256=sha)
        assert second.status_code == 201
        assert second.json()["upload"] is None
        assert second.json()["job"]["status"] == "done"
        assert second.json()["job"]["cached"] is True

    @pytest.mark.parametrize(
        ("body", "status", "message"),
        [
            ({"size": 30 * 1024 * 1024}, 413, "20 MB"),
            ({"filename": "resim.png"}, 415, "PDF"),
            ({"size": 0}, 422, "Geçersiz"),
            ({"sha256": "kisa"}, 422, "Geçersiz"),
        ],
    )
    def test_dogrulama(self, client: TestClient, body, status, message) -> None:
        response = _create(client, **body)
        assert response.status_code == status
        assert message in response.json()["error"]

    def test_gunluk_kota(self, tmp_path, engine) -> None:
        settings = Settings(data_dir=tmp_path, daily_job_limit=2)
        client = TestClient(build_local_app(settings, engine_factory=lambda: engine))
        assert [_create(client).status_code for _ in range(3)] == [201, 201, 429]

    def test_yukleme_kurallari(self, client: TestClient, article_bytes: bytes) -> None:
        job_id = _create(client).json()["job"]["id"]
        url = f"/api/uploads/{job_id}"
        not_pdf = client.post(url, files={"file": ("x.pdf", b"merhaba", "application/pdf")})
        assert not_pdf.status_code == 415
        ok = client.post(url, files={"file": ("x.pdf", article_bytes, "application/pdf")})
        assert ok.status_code == 202
        again = client.post(url, files={"file": ("x.pdf", article_bytes, "application/pdf")})
        assert again.status_code == 409

    def test_buyuk_yukleme_reddedilir(self, tmp_path, engine) -> None:
        settings = Settings(data_dir=tmp_path, max_upload_bytes=100)
        client = TestClient(build_local_app(settings, engine_factory=lambda: engine))
        job_id = client.post("/api/jobs", json={"filename": "a.pdf", "size": 50}).json()["job"][
            "id"
        ]
        big = b"%PDF-1.4" + b"0" * 200
        response = client.post(f"/api/uploads/{job_id}", files={"file": ("a.pdf", big)})
        assert response.status_code == 413

    @pytest.mark.parametrize("job_id", ["0" * 32, "kotu-kimlik", "../etc"])
    def test_bulunamadi(self, client: TestClient, job_id: str) -> None:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 404
        assert "bulunamadı" in response.json()["error"]

    def test_arayuz_sunulur(self, tmp_path, engine) -> None:
        (tmp_path / "web").mkdir()
        (tmp_path / "web" / "index.html").write_text("<h1>ozetle</h1>", encoding="utf-8")
        settings = Settings(data_dir=tmp_path, frontend_dir=tmp_path / "web")
        client = TestClient(build_local_app(settings, engine_factory=lambda: engine))
        assert "ozetle" in client.get("/").text
        assert client.get("/api/health").status_code == 200

    def test_model_yuklenemezse_is_basarisiz(self, tmp_path, article_bytes) -> None:
        def bozuk() -> Engine:
            raise ModelLoadError("ağırlık eksik")

        settings = Settings(data_dir=tmp_path)
        client = TestClient(build_local_app(settings, engine_factory=bozuk, synchronous=True))
        created = client.post("/api/jobs", json={"filename": "a.pdf", "size": 10}).json()
        client.post(created["upload"]["url"], files={"file": ("a.pdf", article_bytes)})
        job = client.get(f"/api/jobs/{created['job']['id']}").json()
        assert job["status"] == "failed"
        assert "Model yüklenemedi" in job["error"]

    def test_arka_plan_havuzu(self, tmp_path, engine, article_bytes) -> None:
        settings = Settings(data_dir=tmp_path)
        client = TestClient(build_local_app(settings, engine_factory=lambda: engine))
        created = client.post("/api/jobs", json={"filename": "a.pdf", "size": 10}).json()
        client.post(created["upload"]["url"], files={"file": ("a.pdf", article_bytes)})
        for _ in range(100):
            status = client.get(f"/api/jobs/{created['job']['id']}").json()["status"]
            if status == "done":
                break
            time.sleep(0.05)
        assert status == "done"


# ---------------------------------------------------------------------------
# Lambda giriş noktaları
# ---------------------------------------------------------------------------


def _http_event(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """API Gateway HTTP API (payload 2.0) olayı."""
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"content-type": "application/json", "host": "api.example.com"},
        "requestContext": {
            "http": {"method": method, "path": path, "sourceIp": "1.2.3.4", "protocol": "HTTP/1.1"},
            "requestId": "r",
            "stage": "$default",
        },
        "body": json.dumps(body) if body is not None else None,
        "isBase64Encoded": False,
    }


def _aws_settings() -> Settings:
    return Settings(backend="aws", table_name=TABLE, upload_bucket=BUCKET)


class TestLambdaApi:
    def test_is_olustur_s3_hedefi(self, aws) -> None:
        from mangum import Mangum

        app = lambda_api.build_app(_aws_settings(), dynamodb=aws["dynamodb"], s3=aws["s3"])
        handler = Mangum(app, lifespan="off")
        response = handler(
            _http_event("POST", "/api/jobs", {"filename": "a.pdf", "size": 100}), None
        )
        assert response["statusCode"] == 201
        body = json.loads(response["body"])
        assert BUCKET in body["upload"]["url"]
        assert body["upload"]["fields"]["key"] == f"uploads/{body['job']['id']}.pdf"
        stored = DynamoJobStore(TABLE, aws["dynamodb"]).get(body["job"]["id"])
        assert stored is not None and stored.status is JobStatus.AWAITING_UPLOAD
        # AWS'de local yükleme uç noktası yoktur; dosya doğrudan S3'e gider.
        upload = handler(_http_event("POST", f"/api/uploads/{body['job']['id']}"), None)
        assert upload["statusCode"] in (404, 405)

    def test_handler_tembel_kurulur(self, aws, monkeypatch) -> None:
        monkeypatch.setenv("OZETLE_TABLE_NAME", TABLE)
        monkeypatch.setenv("OZETLE_UPLOAD_BUCKET", BUCKET)
        monkeypatch.setattr(lambda_api, "_handler", None)
        response = lambda_api.handler(_http_event("GET", "/api/health"), None)
        assert json.loads(response["body"])["backend"] == "aws"


def _s3_message(job_id: str, message_id: str = "m1") -> dict[str, Any]:
    body = {"Records": [{"s3": {"object": {"key": f"uploads/{job_id}.pdf"}}}]}
    return {"messageId": message_id, "body": json.dumps(body)}


class TestLambdaWorker:
    def _deps(self, aws, engine_factory) -> lambda_worker.WorkerDeps:
        return lambda_worker.WorkerDeps(
            settings=_aws_settings(),
            store=DynamoJobStore(TABLE, aws["dynamodb"]),
            uploads=S3UploadStore(BUCKET, aws["s3"]),
            engine_factory=engine_factory,
        )

    def test_s3_olayindan_is(self, aws, engine, article_bytes) -> None:
        deps = self._deps(aws, lambda: engine)
        job = _job()
        deps.store.create(job)
        aws["s3"].put_object(Bucket=BUCKET, Key=f"uploads/{job.id}.pdf", Body=article_bytes)

        response = lambda_worker.handle({"Records": [_s3_message(job.id)]}, deps)

        assert response == {"batchItemFailures": []}
        done = deps.store.get(job.id)
        assert done is not None and done.status is JobStatus.DONE
        assert aws["s3"].list_objects_v2(Bucket=BUCKET).get("KeyCount", 0) == 0

    def test_test_olayi_atlanir(self, aws, engine) -> None:
        deps = self._deps(aws, lambda: engine)
        record = {"messageId": "t", "body": json.dumps({"Event": "s3:TestEvent"})}
        assert lambda_worker.handle({"Records": [record]}, deps) == {"batchItemFailures": []}

    def test_kismi_toplu_hata(self, aws, article_bytes) -> None:
        """Yalnızca başarısız mesaj yeniden denenir."""

        def bozuk() -> Engine:
            raise ModelLoadError("bellek yetersiz")

        deps = self._deps(aws, bozuk)
        job = _job()
        deps.store.create(job)
        aws["s3"].put_object(Bucket=BUCKET, Key=f"uploads/{job.id}.pdf", Body=article_bytes)
        garbage = {"messageId": "bozuk", "body": "{json degil"}
        response = lambda_worker.handle(
            {"Records": [_s3_message(job.id, "m1"), garbage, _s3_message("0" * 32, "m3")]}, deps
        )
        assert response == {
            "batchItemFailures": [{"itemIdentifier": "m1"}, {"itemIdentifier": "bozuk"}]
        }

    def test_handler_bagimliliklari_bir_kez_kurar(self, monkeypatch) -> None:
        built: list[int] = []
        sentinel = object()

        def fake_build() -> Any:
            built.append(1)
            return sentinel

        monkeypatch.setattr(lambda_worker, "_deps", None)
        monkeypatch.setattr(lambda_worker, "build_deps", fake_build)
        monkeypatch.setattr(lambda_worker, "handle", lambda event, deps: {"deps": deps})
        assert lambda_worker.handler({}, None) == {"deps": sentinel}
        lambda_worker.handler({}, None)
        assert built == [1]


# ---------------------------------------------------------------------------
# Loglama
# ---------------------------------------------------------------------------


class TestLogs:
    def test_json_satiri(self) -> None:
        record = logging.LogRecord("ozetle", logging.INFO, "f", 1, "İş %s", ("bitti",), None)
        record.job_id = "abc"
        line = json.loads(JsonFormatter().format(record))
        assert line["msg"] == "İş bitti"
        assert line["job_id"] == "abc"
        assert line["level"] == "INFO"

    def test_istisna_eklenir(self) -> None:
        try:
            raise ValueError("patladı")
        except ValueError:
            import sys

            record = logging.LogRecord("x", logging.ERROR, "f", 1, "hata", None, sys.exc_info())
        assert "patladı" in json.loads(JsonFormatter().format(record))["exc"]

    def test_mevcut_handler_yeniden_kullanilir(self) -> None:
        root = logging.getLogger()
        before = list(root.handlers)
        configure_logging("INFO")
        configure_logging("INFO")
        assert len(root.handlers) == max(1, len(before))
        assert all(isinstance(h.formatter, JsonFormatter) for h in root.handlers)
        for handler in root.handlers:
            handler.setFormatter(None)


class TestEngine:
    def test_motor_tembel_kurulur(self, monkeypatch) -> None:
        import importlib.util

        from ozetle.models.summarizer import HFSummarizer
        from ozetle.service import engine as engine_module
        from ozetle.service.engine import build_engine, cached_engine_factory

        # keybert kurulu olmasa da (CI test işi) KeyBERT yolunu sına; kurulum
        # tembel olduğu için paket gerçekten içe aktarılmaz.
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: object() if name == "keybert" else real_find_spec(name, *a),
        )
        engine_module._engine_for.cache_clear()

        engine = build_engine(Settings(model="mt5-base", keywords_enabled=False))
        assert isinstance(engine.summarizer, HFSummarizer)
        assert not engine.summarizer.loaded  # kurulum indirme yapmaz
        assert engine.keywords is None
        factory = cached_engine_factory(Settings(model="mt5-base"))
        assert factory() is factory()  # süreç başına tek motor
        assert factory().keywords is not None

    def test_lead_modu_model_indirmez(self, monkeypatch) -> None:
        import importlib.util

        from ozetle.models.summarizer import LeadSummarizer
        from ozetle.service.engine import build_engine

        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: None if name == "keybert" else real_find_spec(name, *a),
        )
        engine = build_engine(Settings(model="lead"))
        assert isinstance(engine.summarizer, LeadSummarizer)
        assert engine.keywords is None  # keybert yok → kapalı, çökme yok
