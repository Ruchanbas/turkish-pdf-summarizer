"""Local sunucu: ``python -m ozetle.service`` → http://localhost:8000"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - uvicorn döngüsü
    import uvicorn

    from ozetle.service.config import Settings
    from ozetle.service.local import build_local_app
    from ozetle.service.logs import configure_logging

    parser = argparse.ArgumentParser(prog="python -m ozetle.service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--frontend", type=Path, default=Path("frontend"), help="Arayüz klasörü (yoksa sunulmaz)"
    )
    args = parser.parse_args(argv)

    settings = Settings(backend="local")
    if settings.frontend_dir is None and args.frontend.is_dir():
        settings = settings.model_copy(update={"frontend_dir": args.frontend})
    configure_logging(settings.log_level)
    uvicorn.run(build_local_app(settings), host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":  # pragma: no cover
    main()
