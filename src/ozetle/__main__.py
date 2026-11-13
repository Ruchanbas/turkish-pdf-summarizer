"""Komut satırından hattı denemek için: ``python -m ozetle belge.pdf``.

Model gerektirmez. PDF'in hangi aşamada ne kadar küçüldüğünü ve modele
kaç parça gideceğini gösterir.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from ozetle.pdf.extract import PdfError
from ozetle.pipeline import prepare
from ozetle.text.chunk import DEFAULT_MAX_TOKENS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ozetle",
        description="Türkçe PDF'i özetleme modeline hazır parçalara böler.",
    )
    parser.add_argument("pdf", help="PDF dosya yolu")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-columns", action="store_true", help="Sütun tespitini kapat")
    parser.add_argument("--show-chunks", action="store_true", help="Parçaların metnini yazdır")
    parser.add_argument("--json", action="store_true", help="Çıktıyı JSON olarak ver")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        doc = prepare(args.pdf, max_tokens=args.max_tokens, detect_columns=not args.no_columns)
    except PdfError as exc:
        print(f"hata: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "pages": {"total": doc.total_pages, "read": doc.pages_read},
            "metrics": doc.metrics.as_dict(),
            "garbled_ratio": round(doc.garbled_ratio, 4),
            "warnings": doc.warnings,
            "chunks": [
                {"index": c.index, "tokens": c.token_count, "text": c.text} for c in doc.chunks
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    m = doc.metrics
    print(f"Sayfa          : {doc.pages_read}/{doc.total_pages}")
    print(f"Ham metin      : {m.raw:>7} karakter")
    print(f"Süsleme sonrası: {m.after_furniture:>7}  (-{m.furniture_removed})")
    print(f"Kaynakça sonra : {m.after_references:>7}  (-{m.references_removed})")
    print(f"Temiz metin    : {m.after_clean:>7}")
    print(f"Parça          : {len(doc.chunks)}  (bütçe {args.max_tokens} token)")
    for warning in doc.warnings:
        print(f"uyarı: {warning}")
    if args.show_chunks:
        for chunk in doc.chunks:
            print(f"\n--- parça {chunk.index} · {chunk.token_count} token ---\n{chunk.text}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
