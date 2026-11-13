"""PDF'ten çıkan ham metni özetlemeye hazır hale getirir.

Tasarım ilkesi: **temizlik geri döndürülemez, bu yüzden muhafazakâr olmalı.**
Bir kelimeyi yanlışlıkla silmek, bir gürültüyü bırakmaktan pahalıdır \u2014 çünkü
silinen bilgi özete asla giremez, bırakılan gürültüyü model çoğu zaman
zaten görmezden gelir.

Bu modül bilinçli olarak şunları YAPMAZ:
    - Belirli kelimeleri kör silme ("Tablo", "Şekil" gibi)
    - İçinde sayı geçen her parantezi uçurma (yıllar, dozlar, p-değerleri gider)
    - Stopword ayıklama (özetleme modeli bağlama ihtiyaç duyar)
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

#: PDF gömme fontlarından gelen ligatürler. pdfplumber bunları tek karakter
#: olarak döndürür ve tokenizer'lar bunlar için <unk> üretir.
LIGATURES: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

#: Görünmez ama tokenizer'ı bozan karakterler.
_INVISIBLE = re.compile("[­​‌‍⁠﻿]")

#: Satır kırılmayan boşluk ve benzerleri — düz boşluğa çevrilir.
_ODD_SPACES = re.compile("[\u00a0\u2002-\u200a\u202f\u205f\u3000\t]")

#: "İ".lower() Python'da "i̇" (i + U+0307) üretir. Bazı PDF üreticileri bu
#: ayrışık biçimi metne gömer; NFC bunu birleştiremez çünkü öncül karakteri
#: yoktur. Görünüşte "i" olan ama karşılaştırmada eşleşmeyen kelimeler doğar.
_COMBINING_DOT_I = re.compile("i̇")

#: Satır sonu tirelemesi: "değerlendir-\nmesi" → "değerlendirmesi".
#: Yalnızca sonraki satır küçük harfle başlıyorsa birleştirilir; büyük harf
#: veya rakamla başlıyorsa tire büyük ihtimalle gerçektir ("COVID-\n19").
_HYPHEN_BREAK = re.compile(r"(\w)-\n[ \t]*(?=[a-zçğıöşü])")

#: Numaralı atıf işaretleri: [3], [1, 2], [4-7], [4-7], [1,3-5].
#: Yalnızca köşeli parantez — yuvarlak parantez içeriği (yıl, doz, p-değeri)
#: bilinçli olarak dokunulmadan bırakılır.
_NUMERIC_CITATION = re.compile(r"\s?\[\d{1,3}(?:\s*[,;\-\u2013]\s*\d{1,3})*\]")

_SPACE_BEFORE_PUNCT = re.compile(r"[ ]+([.,;:!?)\]»])")
_MULTI_SPACE = re.compile(r"[ ]{2,}")
_PARAGRAPH_BREAK = re.compile(r"\n[ ]*\n+")


# ---------------------------------------------------------------------------
# Adımlar
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """Ligatürleri açar, görünmez karakterleri atar, NFC'ye getirir.

    >>> normalize_unicode("e\\ufb03cient")
    'efficient'
    """
    for lig, plain in LIGATURES.items():
        text = text.replace(lig, plain)
    text = _INVISIBLE.sub("", text)
    text = _ODD_SPACES.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    return _COMBINING_DOT_I.sub("i", text)


def turkish_lower(text: str) -> str:
    """Türkçe kurallarıyla küçük harfe çevirir.

    Python'un ``str.lower()`` metodu dilden bağımsızdır: "I" → "i" (Türkçede
    "ı" olmalı) ve "İ" → "i̇" (i + U+0307 birleşik nokta) üretir. İkincisi
    özellikle sinsidir: ekranda "i" görünür ama "istanbul" ile eşleşmez,
    bu yüzden stopword listeleri ve tekrar ayıklama sessizce çalışmaz.

    >>> turkish_lower("İSTANBUL ILIK")
    'istanbul ılık'
    """
    return text.replace("I", "ı").replace("İ", "i").lower()


def dehyphenate(text: str) -> str:
    """Satır sonunda bölünmüş kelimeleri birleştirir.

    Satır yapısı hâlâ dururken çağrılmalıdır; satırlar birleştikten sonra
    hangi tirenin kırılma tiresi olduğu anlaşılamaz.

    >>> dehyphenate("değerlendir-\\nmesi yapıldı")
    'değerlendirmesi yapıldı'
    """
    return _HYPHEN_BREAK.sub(r"\1", text)


def remove_citation_markers(text: str) -> str:
    """Köşeli parantezli numaralı atıfları siler.

    "(Demir ve ark., 2019)" gibi yazar-yıl atıfları **silinmez**: yuvarlak
    parantezi sayıya göre ayırt etmenin güvenli yolu yoktur ve yanlış silme
    "(n = 45)" veya "(p < 0.05)" gibi bulguları yok eder.

    >>> remove_citation_markers("Etkili bulundu [3, 4].")
    'Etkili bulundu.'
    """
    return _NUMERIC_CITATION.sub("", text)


def unwrap_lines(text: str) -> str:
    """PDF'in görsel satır kırılmalarını kaldırır, paragrafları korur.

    Tek satır sonu → boşluk. Boş satır → paragraf sınırı (``\\n\\n``).
    Parçalayıcı paragraf sınırını cümle sınırı olarak kullanır.
    """
    paragraphs = _PARAGRAPH_BREAK.split(text)
    joined = (" ".join(line.strip() for line in p.splitlines()) for p in paragraphs)
    return "\n\n".join(p for p in joined if p.strip())


def normalize_whitespace(text: str) -> str:
    """Fazla boşlukları ve atıf silmenin bıraktığı izleri toparlar."""
    lines = []
    for line in text.split("\n"):
        line = _MULTI_SPACE.sub(" ", line)
        line = _SPACE_BEFORE_PUNCT.sub(r"\1", line)
        lines.append(line.strip())
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def clean(text: str, *, remove_citations: bool = True) -> str:
    """Ham PDF metnini özetleme modeline hazırlar.

    Sıra önemlidir: tire birleştirme satır yapısına ihtiyaç duyar, bu yüzden
    satırlar açılmadan önce yapılır.

    Args:
        text: PDF'ten çıkarılmış, sayfa süslemesi ve kaynakçası atılmış metin.
        remove_citations: Köşeli parantezli atıf işaretleri silinsin mi.

    Returns:
        Paragrafları ``\\n\\n`` ile ayrılmış temiz metin.
    """
    if not text:
        return ""
    text = normalize_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = dehyphenate(text)
    if remove_citations:
        text = remove_citation_markers(text)
    text = unwrap_lines(text)
    return normalize_whitespace(text)
