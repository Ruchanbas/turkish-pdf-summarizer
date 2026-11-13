"""PDF'ten sayfa sayfa metin çıkarır.

pdfplumber (MIT) kullanılır. PyMuPDF daha hızlıdır ama AGPL lisanslıdır;
bir servis olarak dağıtılacak projenin lisansını kirletmemesi için tercih
edilmedi.

Bu katman yalnızca okur ve ölçer, hiçbir şeyi yorumlamaz: süsleme
temizliği :mod:`ozetle.pdf.layout`'ta, metin temizliği
:mod:`ozetle.text.clean`'de yapılır. Sayfa yapısı burada bilinçli olarak
korunur çünkü sonraki adımlar ona ihtiyaç duyar.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
from pdfminer.pdfdocument import PDFEncryptionError, PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.psparser import PSException
from pdfplumber.utils.exceptions import MalformedPDFException, PdfminerException

from ozetle.limits import MAX_FILE_BYTES, MAX_PAGES

logger = logging.getLogger(__name__)

_LINE_TOLERANCE = 3.0
"""Aynı satır sayılmak için dikey tolerans (punto)."""

_CID_ARTIFACT = re.compile(r"\(cid:\d+\)")
"""pdfplumber, gömülü olmayan fontlarda eşleşmeyen glifleri "(cid:123)"
olarak döndürür. Bu, kodlamanın bozuk olduğunun kesin işaretidir."""

_GARBLE_THRESHOLD = 0.02
"""Karakterlerin bu oranı kayıpsa kodlama sorunu bildirilir."""

# Sütun tespiti parametreleri
_GUTTER_SEARCH = (0.35, 0.65)
"""Sütun boşluğunun aranacağı yatay bant (sayfa genişliğine oranla)."""
_MIN_COLUMN_SHARE = 0.25
"""Her sütunun kelimelerin en az bu kadarını taşıması gerekir."""
_MAX_SPANNING_SHARE = 0.15
"""Boşluğu kesen (başlık gibi tam genişlik) kelimelerin azami oranı."""
_MIN_VERTICAL_OVERLAP = 0.5
"""İki sütunun dikeyde örtüşme oranı — yan yana durduklarının kanıtı."""
_MIN_GUTTER_WIDTH = 8.0
"""Sütun boşluğunun asgari genişliği (punto). Sıradan bir kelime arası
9-10 puntoluk metinde 2-3 punto kadardır."""


# ---------------------------------------------------------------------------
# Hatalar
# ---------------------------------------------------------------------------


class PdfError(Exception):
    """PDF okuma katmanının tüm hatalarının tabanı.

    API katmanı bu sınıfları HTTP koduna çevirir; mesajlar kullanıcıya
    gösterilebilecek şekilde yazılır.
    """


class InvalidPdfError(PdfError):
    """Dosya PDF değil veya bozuk."""


class EncryptedPdfError(PdfError):
    """PDF parola korumalı."""


class PdfTooLargeError(PdfError):
    """Dosya boyut sınırını aşıyor."""


class EmptyPdfError(PdfError):
    """PDF'te çıkarılabilir metin yok — büyük ihtimalle taranmış görüntü."""


# ---------------------------------------------------------------------------
# Veri tipleri
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PdfPage:
    number: int
    """1'den başlayan sayfa numarası."""
    text: str
    columns: int = 1
    """Tespit edilen sütun sayısı (1 veya 2)."""


@dataclass(frozen=True)
class PdfDocument:
    pages: list[PdfPage]
    total_pages: int
    truncated: bool = False
    """Sayfa sınırı yüzünden belgenin bir kısmı okunmadı mı."""
    metadata: dict[str, str] = field(default_factory=dict)
    garbled_ratio: float = 0.0
    """Kodlama bozukluğu oranı. 0.02 üstü ciddi sorun demektir."""

    @property
    def has_encoding_issues(self) -> bool:
        """Metin kodlaması güvenilmez mi.

        Hata değil uyarıdır: kullanıcının işi engellenmez ama çıktının
        neden kötü olabileceği kendisine söylenir.
        """
        return self.garbled_ratio > _GARBLE_THRESHOLD

    @property
    def pages_read(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        """Sayfaların ``\\n`` ile birleştirilmiş halinin uzunluğu.

        Hat metrikleri bu tanımı kullanır; her aşama aynı biçimde
        (birleştirilmiş metin üzerinden) ölçülmelidir.
        """
        return len("\n".join(self.texts()))

    def texts(self) -> list[str]:
        return [p.text for p in self.pages]


# ---------------------------------------------------------------------------
# Kodlama kalitesi
# ---------------------------------------------------------------------------


def measure_garbling(text: str) -> float:
    """Kodlama bozukluğu oranını ölçer (0.0 = temiz).

    Gömülü olmayan veya özel kodlamalı fontlar, çıkarımda kayıp karakter
    bırakır. Türkçe bundan orantısız etkilenir: ı, ş, ğ, İ genelde ilk
    kaybolanlardır ve "yapılmıştır" gibi bir kelime "yapnlmnntnr" olur.

    Bu tespit edilmezse model bozuk metni özetler ve kullanıcı neden
    saçma bir çıktı aldığını asla anlayamaz.
    """
    if not text:
        return 0.0
    lost = len(_CID_ARTIFACT.findall(text)) * 8  # "(cid:NN)" ≈ 8 karakter yer kaplar
    lost += text.count("�")
    return min(1.0, lost / len(text))


def strip_cid_artifacts(text: str) -> str:
    """Çıkarılamayan glif işaretlerini temizler.

    İşaretler metinde kalırsa tokenizer bunları içerik sanır. Silmek
    bilgiyi geri getirmez ama gürültüyü azaltır.
    """
    return _CID_ARTIFACT.sub("", text).replace("�", "")


# ---------------------------------------------------------------------------
# Satır kurma ve sütun tespiti
# ---------------------------------------------------------------------------

Word = dict[str, Any]


def _rows(words: Sequence[Word]) -> list[list[Word]]:
    """Kelimeleri dikey konuma göre satırlara gruplar; satır içi x sırasıyla."""
    rows: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (float(w["top"]), float(w["x0"]))):
        if rows and abs(float(word["top"]) - float(rows[-1][0]["top"])) <= _LINE_TOLERANCE:
            rows[-1].append(word)
        else:
            rows.append([word])
    return [sorted(row, key=lambda w: float(w["x0"])) for row in rows]


def _join(row: Sequence[Word]) -> str:
    return " ".join(str(w["text"]) for w in row)


def _split_row(row: Sequence[Word], gutter: float) -> tuple[list[Word], list[Word]] | None:
    """Satırı sütun boşluğundan ikiye ayırır; satır tam genişlikse ``None``.

    Tam genişlik satır (başlık gibi): boşluğu kesen bir kelime var, ya da
    iki taraftaki kelimeler arasındaki aralık sıradan bir kelime arası
    kadar dar — yani bu satırda gerçek bir sütun boşluğu yok.
    """
    left = [w for w in row if float(w["x1"]) <= gutter]
    right = [w for w in row if float(w["x0"]) >= gutter]
    if len(left) + len(right) < len(row):
        return None
    if left and right:
        gap = min(float(w["x0"]) for w in right) - max(float(w["x1"]) for w in left)
        if gap < _MIN_GUTTER_WIDTH:
            return None
    return left, right


def find_gutter(words: Sequence[Word], page_width: float) -> float | None:
    """İki sütunlu düzende sütunlar arasındaki boşluğun x konumunu bulur.

    Sayfanın orta bandında, sütunları ayıran **geniş** boş şerit aranır.
    Genişlik şartı kritik: benzer satırlardan oluşan tek sütunlu bir
    sayfada kelime araları alt alta hizalanabilir ve hiçbir kelimenin
    kesmediği dar bir çizgi oluşur. O çizgi bir kelime arası kadar dardır;
    gerçek sütun boşluğu ise birkaç kat geniştir.

    Bir şerit iki sütun düzeni sayılır ancak:
        - şerit yeterince genişse,
        - her iki tarafta yeterince kelime varsa,
        - şeridi kesen kelimeler (tam genişlik başlıklar) azsa,
        - iki taraf dikeyde örtüşüyorsa (alt alta değil, yan yana).

    Returns:
        Şeridin orta noktasının x konumu, tek sütunlu sayfada ``None``.
    """
    if len(words) < 20 or page_width <= 0:
        return None

    lo, hi = (page_width * r for r in _GUTTER_SEARCH)
    center = page_width / 2
    best: tuple[float, float] | None = None  # (genişlik, x)

    x = lo
    while x <= hi:
        crossing = sum(1 for w in words if float(w["x0"]) < x < float(w["x1"]))
        if crossing <= _MAX_SPANNING_SHARE * len(words):
            # Tam genişlik satırlar şeridi daraltmasın: yalnızca boşluğu
            # kesmeyen kelimelerle ölç.
            left_edge = max((float(w["x1"]) for w in words if float(w["x1"]) <= x), default=None)
            right_edge = min((float(w["x0"]) for w in words if float(w["x0"]) >= x), default=None)
            if left_edge is not None and right_edge is not None:
                width = right_edge - left_edge
                mid = (left_edge + right_edge) / 2
                if (
                    best is None
                    or width > best[0]
                    or (width == best[0] and abs(mid - center) < abs(best[1] - center))
                ):
                    best = (width, mid)
        x += 2.0

    if best is None or best[0] < _MIN_GUTTER_WIDTH:
        return None
    gutter = best[1]

    left = [w for w in words if float(w["x1"]) <= gutter]
    right = [w for w in words if float(w["x0"]) >= gutter]
    if min(len(left), len(right)) < _MIN_COLUMN_SHARE * len(words):
        return None

    def span(ws: Sequence[Word]) -> tuple[float, float]:
        return min(float(w["top"]) for w in ws), max(float(w["bottom"]) for w in ws)

    (l_top, l_bot), (r_top, r_bot) = span(left), span(right)
    overlap = min(l_bot, r_bot) - max(l_top, r_top)
    shorter = min(l_bot - l_top, r_bot - r_top)
    if shorter <= 0 or overlap < _MIN_VERTICAL_OVERLAP * shorter:
        return None
    return gutter


def words_to_text(
    words: Sequence[Word], page_width: float, *, detect_columns: bool
) -> tuple[str, int]:
    """Kelime kutularından okuma sırasına uygun sayfa metni kurar.

    İki sütunlu sayfada okuma sırası: tam genişlik satırlar yerinde kalır
    (başlık üstte, dipnot altta); aralarındaki her iki sütunlu blokta önce
    sol sütun, sonra sağ sütun okunur. Naif soldan sağa okuma, iki sütunun
    satırlarını birbirine geçirir ve her cümleyi anlamsız hale getirir.

    Returns:
        (sayfa metni, sütun sayısı)
    """
    rows = _rows(words)
    gutter = find_gutter(words, page_width) if detect_columns else None
    if gutter is None:
        return "\n".join(_join(row) for row in rows), 1

    out: list[str] = []
    block_left: list[str] = []
    block_right: list[str] = []

    def flush() -> None:
        out.extend(block_left)
        out.extend(block_right)
        block_left.clear()
        block_right.clear()

    for row in rows:
        split = _split_row(row, gutter)
        if split is None:
            flush()
            out.append(_join(row))
            continue
        left, right = split
        if left:
            block_left.append(_join(left))
        if right:
            block_right.append(_join(right))
    flush()
    return "\n".join(out), 2


def _page_text(page: Any, *, detect_columns: bool) -> tuple[str, int]:
    words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
    return words_to_text(words, float(page.width), detect_columns=detect_columns)


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------


def _open_source(source: str | Path | bytes, max_bytes: int) -> io.BytesIO:
    if isinstance(source, bytes):
        data = source
    else:
        path = Path(source)
        if not path.is_file():
            raise InvalidPdfError(f"Dosya bulunamadı: {path.name}")
        if path.stat().st_size > max_bytes:
            raise PdfTooLargeError(f"Dosya {max_bytes // (1024 * 1024)} MB sınırını aşıyor.")
        data = path.read_bytes()

    if len(data) > max_bytes:
        raise PdfTooLargeError(f"Dosya {max_bytes // (1024 * 1024)} MB sınırını aşıyor.")
    # Bazı üreticiler başlıktan önce birkaç bayt çöp bırakır; spesifikasyon
    # ilk 1024 baytta aranmasına izin verir.
    if b"%PDF-" not in data[:1024]:
        raise InvalidPdfError("Dosya bir PDF değil.")
    return io.BytesIO(data)


def extract(
    source: str | Path | bytes,
    *,
    max_pages: int = MAX_PAGES,
    max_bytes: int = MAX_FILE_BYTES,
    detect_columns: bool = True,
) -> PdfDocument:
    """PDF'ten sayfa sayfa metin çıkarır.

    Args:
        source: Dosya yolu veya PDF baytları (API yüklemesi için).
        max_pages: Okunacak azami sayfa; fazlası kesilir ve işaretlenir.
        max_bytes: Kabul edilen azami boyut.
        detect_columns: İki sütun tespiti yapılsın mı.

    Raises:
        InvalidPdfError: Dosya yok, PDF değil veya bozuk.
        EncryptedPdfError: Parola korumalı.
        PdfTooLargeError: Boyut sınırı aşıldı.
        EmptyPdfError: Çıkarılabilir metin yok (taranmış belge).
    """
    if max_pages <= 0:
        raise ValueError("max_pages pozitif olmalı")

    stream = _open_source(source, max_bytes)
    try:
        with pdfplumber.open(stream) as pdf:
            total = len(pdf.pages)
            pages: list[PdfPage] = []
            for number, page in enumerate(pdf.pages[:max_pages], start=1):
                text, columns = _page_text(page, detect_columns=detect_columns)
                pages.append(PdfPage(number=number, text=text, columns=columns))
                page.close()  # pdfplumber sayfa önbelleğini bırakır; uzun belgede bellek şişmez.
            metadata = {
                str(k): str(v) for k, v in (pdf.metadata or {}).items() if isinstance(v, str)
            }
    except PdfminerException as exc:
        # pdfplumber, pdfminer hatalarını sarmalar; asıl sebep args[0]'da.
        cause = exc.args[0] if exc.args else exc
        if isinstance(cause, (PDFPasswordIncorrect, PDFEncryptionError)):
            raise EncryptedPdfError("PDF parola korumalı.") from exc
        raise InvalidPdfError("PDF okunamadı; dosya bozuk olabilir.") from exc
    except (MalformedPDFException, PDFSyntaxError, PSException) as exc:
        raise InvalidPdfError("PDF okunamadı; dosya bozuk olabilir.") from exc

    raw = "\n".join(p.text for p in pages)
    garbling = measure_garbling(raw)
    if garbling > 0:
        pages = [
            PdfPage(number=p.number, text=strip_cid_artifacts(p.text), columns=p.columns)
            for p in pages
        ]

    if not any(p.text.strip() for p in pages):
        raise EmptyPdfError(
            "PDF'te okunabilir metin bulunamadı. Taranmış bir belge olabilir; "
            "OCR henüz desteklenmiyor."
        )

    document = PdfDocument(
        pages=pages,
        total_pages=total,
        truncated=total > len(pages),
        metadata=metadata,
        garbled_ratio=garbling,
    )

    if document.truncated:
        logger.info("Belge %d sayfa; ilk %d sayfa okundu.", total, len(pages))
    if document.has_encoding_issues:
        logger.warning(
            "PDF kodlaması bozuk görünüyor (oran %.1f%%). Fontlar gömülü olmayabilir; "
            "özet kalitesi düşecektir.",
            garbling * 100,
        )
    return document
