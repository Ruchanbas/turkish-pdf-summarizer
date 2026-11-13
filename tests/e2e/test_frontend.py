"""Arayüzün uçtan uca testi: gerçek tarayıcı, gerçek API, sahte model.

Local sunucu bir iş parçacığında başlar; Playwright PDF'i yükler ve sonucu
okur. Playwright veya bir Chromium yoksa atlanır (CI'da ``e2e`` işi kurar).
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

from conftest import FakeKeywords  # noqa: E402
from ozetle.models.summarizer import LeadSummarizer  # noqa: E402
from ozetle.service.config import Settings  # noqa: E402
from ozetle.service.local import build_local_app  # noqa: E402
from ozetle.service.processing import Engine  # noqa: E402
from pdf_fixtures import make_article_pdf  # noqa: E402

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    import uvicorn

    engine = Engine(summarizer=LeadSummarizer(), keywords=FakeKeywords())
    settings = Settings(
        data_dir=tmp_path_factory.mktemp("data"), frontend_dir=FRONTEND, model="lead"
    )
    port = _free_port()
    config = uvicorn.Config(
        build_local_app(settings, engine_factory=lambda: engine),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/"
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser() -> Iterator[object]:
    with playwright_sync.sync_playwright() as p:
        for kwargs in ({}, {"executable_path": "/usr/bin/google-chrome"}):
            try:
                b = p.chromium.launch(args=["--no-sandbox"], **kwargs)
                break
            except Exception:
                continue
        else:
            pytest.skip("Chromium bulunamadı")
        yield b
        b.close()


@pytest.fixture
def page(browser, server):  # type: ignore[no-untyped-def]
    context = browser.new_context(viewport={"width": 1100, "height": 900})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(server)
    yield page
    context.close()
    assert errors == [], f"Tarayıcı hataları: {errors}"


@pytest.fixture(scope="module")
def article(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_article_pdf(tmp_path_factory.mktemp("e2e") / "makale.pdf", n_body_pages=4)


def test_yukle_ve_ozet_oku(page, article: Path) -> None:  # type: ignore[no-untyped-def]
    assert "20 MB" in page.locator("#limits").inner_text()
    page.set_input_files("#file-input", str(article))
    page.wait_for_selector("#result:not([hidden])", timeout=30000)

    assert page.locator("#result-title").inner_text() == "makale.pdf"
    summary = page.locator("#summary-text").inner_text()
    assert summary.startswith("Bulgu 1:")
    assert page.locator("#summary-text mark").count() >= 1  # anahtar kelime işaretli
    assert page.locator("#keywords mark").all_inner_texts() == ["fizyoterapi", "fonksiyonel skor"]
    assert page.locator("#section-list > li").count() > 1
    assert "sayfa okundu" in page.locator("#result-facts").inner_text()
    assert "#is=" in page.url


def test_baglanti_isi_geri_getirir(page, article: Path) -> None:  # type: ignore[no-untyped-def]
    """Sayfa yenilenince (veya bağlantı paylaşılınca) sonuç yeniden açılır."""
    page.set_input_files("#file-input", str(article))
    page.wait_for_selector("#result:not([hidden])", timeout=30000)
    url = page.url
    page.goto("about:blank")
    page.goto(url)
    page.wait_for_selector("#result:not([hidden])", timeout=10000)
    assert page.locator("#result-title").inner_text() == "makale.pdf"


def test_pdf_olmayan_dosya(page) -> None:  # type: ignore[no-untyped-def]
    page.set_input_files(
        "#file-input",
        files=[{"name": "notlar.txt", "mimeType": "text/plain", "buffer": b"merhaba"}],
    )
    error = page.locator("#intake-error")
    error.wait_for(state="visible")
    assert "PDF değil" in error.inner_text()


def test_sunucu_reddeder(page) -> None:  # type: ignore[no-untyped-def]
    """Uzantısı .pdf olan ama PDF olmayan dosya sunucuda yakalanır."""
    page.set_input_files(
        "#file-input",
        files=[{"name": "sahte.pdf", "mimeType": "application/pdf", "buffer": b"PDF degil"}],
    )
    error = page.locator("#intake-error")
    error.wait_for(state="visible", timeout=10000)
    assert "PDF değil" in error.inner_text()


def test_yeni_pdf_bos_duruma_doner(page, article: Path) -> None:  # type: ignore[no-untyped-def]
    page.set_input_files("#file-input", str(article))
    page.wait_for_selector("#result:not([hidden])", timeout=30000)
    page.click("#again")
    assert page.locator("#intake").is_visible()
    assert "#is=" not in page.url
