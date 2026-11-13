"""PDF'ten modele hazır parçalara giden hat.

Bu modül Faz 1'in tek giriş noktasıdır. Adımlar bilinçli bir sırada:

    çıkar → sayfa süslemesini at → kaynakçayı kes → temizle → parçala

Sıra rastgele değil. Süsleme temizliği **sayfa yapısı hâlâ dururken**
yapılmalı; sayfalar birleştikten sonra hangi satırın header olduğunu
anlamak imkânsızdır. Kaynakça kesimi satır sonları hâlâ anlamlıyken
yapılmalı; satırlar birleştikten sonra başlık tespiti bozulur. Temizlik
en sonda, çünkü satır yapısını bilinçli olarak yok eder.

Ayrıca her aşamanın ne yaptığı ölçülür. Bu sayılar ileride /metrics
uç noktasında ve README'deki karşılaştırmalarda kullanılır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ozetle.pdf.extract import MAX_PAGES, PdfDocument, extract
from ozetle.pdf.layout import cut_references, strip_page_furniture
from ozetle.text.chunk import DEFAULT_MAX_TOKENS, Chunk, TokenCounter, chunk_text
from ozetle.text.clean import clean


@dataclass(frozen=True, slots=True)
class StageMetrics:
    """Her aşamadan sonra kalan karakter sayısı.

    Tüm aşamalar **aynı biçimde** ölçülür: sayfalar ``\\n`` ile
    birleştirilmiş metnin uzunluğu. Bir aşamayı birleştirme öncesi, diğerini
    sonrası saymak, kesme adımı karakter eklemiş gibi gösterir — bu hatanın
    regresyon testi ``test_metrikler_monoton_azalir``.
    """

    raw: int
    after_furniture: int
    after_references: int
    after_clean: int

    @property
    def furniture_removed(self) -> int:
        return self.raw - self.after_furniture

    @property
    def references_removed(self) -> int:
        return self.after_furniture - self.after_references

    def as_dict(self) -> dict[str, int]:
        return {
            "raw": self.raw,
            "after_furniture": self.after_furniture,
            "after_references": self.after_references,
            "after_clean": self.after_clean,
        }


@dataclass(frozen=True)
class PreparedDocument:
    """Özetleme modeline verilmeye hazır belge."""

    chunks: list[Chunk]
    text: str
    """Temizlenmiş tam metin (parçalanmadan önce)."""
    metrics: StageMetrics
    total_pages: int
    pages_read: int
    garbled_ratio: float = 0.0
    warnings: list[str] = field(default_factory=list)
    """Kullanıcıya gösterilecek, işi engellemeyen uyarılar."""

    @property
    def is_empty(self) -> bool:
        return not self.chunks


def prepare_document(
    document: PdfDocument,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_sentences: int = 1,
    token_counter: TokenCounter | None = None,
    remove_citations: bool = True,
) -> PreparedDocument:
    """Çıkarılmış bir belgeyi parçalara dönüştürür (PDF okumadan).

    PDF'ten bağımsız test edilebilmesi ve ileride başka kaynaklardan gelen
    metinlerin aynı hattan geçebilmesi için :func:`prepare`'den ayrıdır.
    """
    raw_chars = document.char_count

    # 1. Süsleme temizliği — sayfa yapısı hâlâ duruyorken.
    pages = strip_page_furniture(document.texts())
    joined = "\n".join(pages)
    after_furniture = len(joined)

    # 2. Kaynakça kesimi — satır sonları hâlâ anlamlıyken.
    joined = cut_references(joined)
    after_references = len(joined)

    # 3. Normalizasyon — satır yapısını bilinçli olarak yok eder.
    text = clean(joined, remove_citations=remove_citations)

    # 4. Parçalama.
    chunks = chunk_text(
        text,
        max_tokens=max_tokens,
        overlap_sentences=overlap_sentences,
        token_counter=token_counter,
    )

    warnings: list[str] = []
    if document.truncated:
        warnings.append(
            f"Belge {document.total_pages} sayfa; ilk {document.pages_read} sayfa işlendi."
        )
    if document.has_encoding_issues:
        warnings.append("PDF'in font kodlaması bozuk görünüyor. Özet kalitesi düşük olabilir.")
    if token_counter is None:
        warnings.append("Gerçek tokenizer verilmedi; parça sınırları tahmini.")
    if not chunks:
        warnings.append("Temizlik sonrası özetlenecek metin kalmadı.")

    return PreparedDocument(
        chunks=chunks,
        text=text,
        metrics=StageMetrics(
            raw=raw_chars,
            after_furniture=after_furniture,
            after_references=after_references,
            after_clean=len(text),
        ),
        total_pages=document.total_pages,
        pages_read=document.pages_read,
        garbled_ratio=document.garbled_ratio,
        warnings=warnings,
    )


def prepare(
    source: str | Path | bytes,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_sentences: int = 1,
    max_pages: int = MAX_PAGES,
    token_counter: TokenCounter | None = None,
    remove_citations: bool = True,
    detect_columns: bool = True,
) -> PreparedDocument:
    """PDF'i özetlemeye hazır parçalara dönüştürür.

    Args:
        source: PDF dosya yolu veya baytları.
        max_tokens: Parça başına token bütçesi. Modelin pencere boyutundan
            düşük olmalı.
        overlap_sentences: Parçalar arası örtüşen cümle sayısı.
        max_pages: Okunacak azami sayfa.
        token_counter: Gerçek tokenizer. Verilmezse sezgisel tahmin
            kullanılır ve parça sınırları yaklaşık olur.
        remove_citations: Atıf işaretleri temizlensin mi.
        detect_columns: İki sütun tespiti yapılsın mı.

    Returns:
        Parçalar, temiz metin, aşama metrikleri ve uyarılar.

    Raises:
        PdfError: Çıkarma katmanının hataları olduğu gibi yükselir —
            API katmanı bunları HTTP koduna çevirir.
    """
    document = extract(source, max_pages=max_pages, detect_columns=detect_columns)
    return prepare_document(
        document,
        max_tokens=max_tokens,
        overlap_sentences=overlap_sentences,
        token_counter=token_counter,
        remove_citations=remove_citations,
    )
