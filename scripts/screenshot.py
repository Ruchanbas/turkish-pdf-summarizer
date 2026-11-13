"""README ekran görüntüsü: gerçek modelle local sunucu + gerçek tarayıcı.

Sahte değil: örnek PDF arayüzden yüklenir, gerçek model özetler, sonuç
ekranı fotoğraflanır. Özet ve anahtar kelimeler loga da yazılır.

    python scripts/screenshot.py --model mt5-small --pdf docs/ornek-makale.pdf \\
        --out docs/screenshot.png
"""

from __future__ import annotations

import argparse
import socket
import tempfile
import threading
import time
from pathlib import Path

import uvicorn
from playwright.sync_api import sync_playwright

from ozetle.service.config import Settings
from ozetle.service.local import build_local_app

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mt5-small")
    parser.add_argument("--pdf", type=Path, default=ROOT / "docs" / "ornek-makale.pdf")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "screenshot.png")
    parser.add_argument("--chrome", help="Playwright'ın Chromium'u yerine bir Chrome yolu")
    args = parser.parse_args()

    settings = Settings(
        model=args.model,
        data_dir=Path(tempfile.mkdtemp()),
        frontend_dir=ROOT / "frontend",
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(build_local_app(settings), host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chrome, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1200, "height": 1500}, device_scale_factor=1)
        page.goto(f"http://127.0.0.1:{port}/")
        page.set_input_files("#file-input", str(args.pdf))
        page.wait_for_selector("#result:not([hidden])", timeout=15 * 60 * 1000)
        page.wait_for_timeout(2500)  # fosforlu kalem animasyonu bitsin
        args.out.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(args.out))
        print("ÖZET:", page.locator("#summary-text").inner_text())
        print("ANAHTAR KELİMELER:", page.locator("#keywords mark").all_inner_texts())
        print("BİLGİ:", page.locator("#result-facts").inner_text())
        for i, note in enumerate(page.locator(".section-note").all_inner_texts(), start=1):
            print(f"BÖLÜM {i}:", note)
        browser.close()
    server.should_exit = True
    print(args.out)


if __name__ == "__main__":
    main()
