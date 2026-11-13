"""Testler için gerçekçi Türkçe PDF'ler üretir.

Neden DejaVu: reportlab'in yerleşik Helvetica'sı WinAnsi kodlaması kullanır
ve ``ı ş ğ İ`` taşıyamaz. İlk fixture'lar Helvetica ile yazılmıştı;
"yapılmıştır" PDF'e bozuk giriyor, testler yine de geçiyordu — yani Türkçe
test ettiğimizi sanıyorduk ama etmiyorduk. ``test_helvetica_turkce_tasiyamaz``
bu tuzağı belgeler.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT_PATH = Path(__file__).parent / "fixtures" / "fonts" / "DejaVuSans.ttf"
FONT = "DejaVuSans"

PAGE_W, PAGE_H = A4
MARGIN = 50
LINE_H = 13.0

if FONT not in pdfmetrics.getRegisteredFontNames():
    pdfmetrics.registerFont(TTFont(FONT, str(FONT_PATH)))


def _draw_lines(c: canvas.Canvas, lines: Sequence[str], x: float, y: float) -> float:
    for line in lines:
        c.drawString(x, y, line)
        y -= LINE_H
    return y


def make_pdf(
    path: Path,
    pages: Sequence[Sequence[str]],
    *,
    font: str = FONT,
    size: int = 9,
    title: str | None = None,
    encrypt: str | None = None,
) -> Path:
    """Her sayfası verilen satırlardan oluşan tek sütunlu PDF."""
    c = canvas.Canvas(str(path), pagesize=A4, encrypt=encrypt)
    if title:
        c.setTitle(title)
    for lines in pages:
        c.setFont(font, size)
        _draw_lines(c, lines, MARGIN, PAGE_H - MARGIN)
        c.showPage()
    c.save()
    return path


def make_blank_pdf(path: Path, n_pages: int = 1) -> Path:
    """Metin katmanı olmayan PDF — taranmış belgeyi taklit eder."""
    c = canvas.Canvas(str(path), pagesize=A4)
    for _ in range(n_pages):
        c.rect(MARGIN, MARGIN, 200, 200)  # Görüntü yerine bir şekil.
        c.showPage()
    c.save()
    return path


def make_two_column_pdf(
    path: Path,
    *,
    title: str,
    left: Sequence[str],
    right: Sequence[str],
    size: int = 9,
) -> Path:
    """Üstte tam genişlik başlık, altında iki sütun gövde."""
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont(FONT, 14)
    c.drawString(MARGIN, PAGE_H - MARGIN, title)
    c.setFont(FONT, size)
    top = PAGE_H - MARGIN - 30
    _draw_lines(c, left, MARGIN, top)
    _draw_lines(c, right, PAGE_W / 2 + 15, top)
    c.showPage()
    c.save()
    return path


def wrap(paragraph: str, width: int) -> list[str]:
    return textwrap.wrap(paragraph, width=width)


# ---------------------------------------------------------------------------
# Hazır içerik
# ---------------------------------------------------------------------------

JOURNAL_HEADER = "Türk Fizyoterapi ve Rehabilitasyon Dergisi 2024; 35(2)"


def finding(i: int) -> str:
    """Gerçek bir makaledeki bulgu paragrafına benzeyen, sayı dolu metin.

    Eski temizlik kodu tam da bu tür içeriği siliyordu: parantez içi
    sayılar, "Tablo N", ondalıklar.
    """
    return (
        f"Bulgu {i}: Hastaların fonksiyonel skorlarında anlamlı iyileşme gözlendi "
        f"(p < 0.05). Tablo {i} bu sonuçları özetlemektedir. Ortalama değer "
        f"3.{i}4 olarak ölçüldü ve Dr. Yılmaz'ın önceki çalışmasıyla uyumluydu [{i}]. "
        f"Uygulanan protokol (Demir ve ark., 2019) haftada üç seans olacak şekilde "
        f"planlanmış ve toplam 500 mg doz (n = 45) verilmiştir. Çalışmanın bu "
        f"aşamasında ölçümler iki bağımsız fizyoterapist tarafından yapılmıştır."
    )


def article_pages(n_body_pages: int = 6, *, references: bool = True) -> list[list[str]]:
    """Header, sayfa numarası, gövde ve kaynakça içeren makale sayfaları."""
    pages: list[list[str]] = []
    for p in range(1, n_body_pages + 1):
        body: list[str] = []
        for k in range(3):
            body.extend(wrap(finding((p - 1) * 3 + k + 1), 95))
            body.append("")
        pages.append([JOURNAL_HEADER, "", *body, f"Sayfa {p}"])
    if references:
        refs = [
            f"[{i}] Yazar{i} A. Bir makale başlığı. Dergi Adı. 2019;12:34-56." for i in range(1, 9)
        ]
        pages.append([JOURNAL_HEADER, "", "KAYNAKLAR", *refs, f"Sayfa {n_body_pages + 1}"])
    return pages


def make_article_pdf(path: Path, n_body_pages: int = 6, *, references: bool = True) -> Path:
    return make_pdf(path, article_pages(n_body_pages, references=references), title="Test Makalesi")
