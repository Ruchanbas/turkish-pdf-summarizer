"""Uzun metni modelin token penceresine sığan parçalara böler.

Bu modül projedeki en kritik hatanın çözümü. Önceki sürüm metni doğrudan
512 token'a kesiyordu; 30 sayfalık bir PDF'in ilk paragrafı özetleniyor,
kalan her şey sessizce çöpe gidiyordu.

Strateji: cümle sınırlarından böl, token bütçesine göre paketle, komşu
parçalar arasında az miktarda örtüşme bırak (bağlam kopmasın diye).
Bütçe **sert sınırdır**: hiçbir parça, hiçbir koşulda onu aşmaz.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

TokenCounter = Callable[[str], int]

#: Varsayılan parça bütçesi. mT5/BERT tabanlı özetleyicilerin 512'lik
#: penceresinde özel tokenlar ve güvenlik payı için yer bırakır.
DEFAULT_MAX_TOKENS = 450

#: Türkçe sondan eklemeli; SentencePiece tokenizer'lar kelime başına
#: ortalama 1.4-1.6 token üretir. Tahmin bilerek yüksek tutulur: fazla
#: tahmin parçaları küçültür (zararsız), az tahmin pencereyi taşırır (zararlı).
TOKENS_PER_WORD = 1.6


def heuristic_token_count(text: str) -> int:
    """Gerçek tokenizer yokken kullanılan muhafazakâr token tahmini."""
    words = len(text.split())
    return math.ceil(words * TOKENS_PER_WORD)


# ---------------------------------------------------------------------------
# Türkçe cümle bölme
# ---------------------------------------------------------------------------

#: Sonunda nokta olan ve cümle bitirmeyen kısaltmalar.
#: Bunlar bölünmeden önce koruma altına alınır.
# fmt: off
ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # Unvan
        "Dr", "Doç", "Prof", "Yrd", "Öğr", "Gör", "Arş", "Av", "Sn", "Uz", "Op",
        # Akademik atıf
        # "der" (derleyen) bilinçli olarak yok: "gel der." çok yaygın bir
        # cümle sonudur ve kısaltma sanılırsa iki cümle birleşir.
        "ark", "vd", "bkz", "age", "agm", "çev", "ed", "haz", "yay",
        # Genel
        "vb", "vs", "örn", "yak", "krş", "tsk",
        # Ölçü / referans
        "s", "ss", "sf", "No", "Nu", "md", "bl", "böl", "fık", "Şek", "Çiz",
        "Tab", "Res", "Ort", "Std", "Min", "Maks", "yy",
        # Kurumsal
        "TC", "AŞ", "Ltd", "Şti", "Üniv", "Fak",
    }
)
# fmt: on

#: Cümle başında büyük harfle yazılan biçimler de kısaltmadır ("Bkz. Tablo 3").
#: Büyük harfli biçim yalnızca cümle başında görülür, orada da cümle bitiremez;
#: bu yüzden tüm eşleşmeyi büyük/küçük harf duyarsız yapmaktan güvenlidir
#: ("gel der." cümlesi "der." kısaltmasıyla karışmaz).
_ABBREV_FORMS = ABBREVIATIONS | {a[0].upper() + a[1:] for a in ABBREVIATIONS}

_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _ABBREV_FORMS), key=len, reverse=True)) + r")\."
)

#: Ondalık sayı içindeki noktalar: 3.14, 1.000
_NUMERIC_DOT = re.compile(r"(?<=\d)\.(?=\d)")
#: Sıra sayısı: "2. Bölüm", "15. yüzyıl". Python re değişken genişlikli
#: lookbehind desteklemediği için yakalama grubu kullanılır.
_ORDINAL_DOT = re.compile(r"\b(\d{1,4})\.(?=\s+[a-zçğıöşü])")
#: Baş harf kısaltmaları: "M. Kemal", "A.B.D."
_INITIAL_DOT = re.compile(r"\b([A-ZÇĞİÖŞÜ])\.")

_SENTINEL = "\x00"

#: Cümle sonu: nokta/ünlem/soru (+ kapanış tırnağı/parantezi), ardından
#: boşluk ve büyük harf veya rakam. Kapanış işareti eşleşmeye dahildir ama
#: ``split`` ile tüketilmez — sınır eşleşmenin sonuna konur, böylece
#: '"Bitti."' cümlesi tırnağını kaybetmez.
_SENTENCE_END = re.compile(r"[.!?…][\"'»)\]]*(?=\s+[\"'«(\[]*[A-ZÇĞİÖŞÜ0-9])")


def _protect(text: str) -> str:
    """Cümle bitirmeyen noktaları sentinel ile değiştirir."""
    text = text.replace(_SENTINEL, "")
    text = _ABBREV_PATTERN.sub(lambda m: m.group(1) + _SENTINEL, text)
    text = _NUMERIC_DOT.sub(_SENTINEL, text)
    text = _ORDINAL_DOT.sub(lambda m: m.group(1) + _SENTINEL, text)
    return _INITIAL_DOT.sub(lambda m: m.group(1) + _SENTINEL, text)


def _restore(text: str) -> str:
    return text.replace(_SENTINEL, ".")


def split_sentences(text: str) -> list[str]:
    """Türkçe metni cümlelere böler.

    Kısaltmaları, ondalık sayıları, sıra sayılarını ve baş harfleri
    korur — bunlar naif ``text.split(".")`` yaklaşımının en sık
    tökezlediği yerlerdir.

    >>> split_sentences("Dr. Ayşe geldi. Saat 3.5 idi. Bitti.")
    ['Dr. Ayşe geldi.', 'Saat 3.5 idi.', 'Bitti.']
    """
    if not text or not text.strip():
        return []

    protected = _protect(text)
    sentences: list[str] = []
    # Paragraf sınırları da cümle sınırıdır.
    for paragraph in re.split(r"\n\s*\n", protected):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        start = 0
        for match in _SENTENCE_END.finditer(paragraph):
            sentences.append(_restore(paragraph[start : match.end()].strip()))
            start = match.end()
        tail = paragraph[start:].strip()
        if tail:
            sentences.append(_restore(tail))
    return sentences


# ---------------------------------------------------------------------------
# Parçalama
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    """Modele tek seferde verilecek metin parçası."""

    index: int
    text: str
    token_count: int
    first_sentence: int
    """Parçadaki ilk cümlenin, belgedeki cümle listesindeki sırası."""
    last_sentence: int
    """Parçadaki son cümlenin sırası (dahil)."""


def _split_oversized(sentence: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """Tek başına bütçeyi aşan cümleyi kelime sınırlarından böler.

    Nadirdir (tablo satırları, noktasız uzun listeler) ama olur; bu durumda
    cümle bütünlüğü bütçeye feda edilir — tersi pencereyi taşırır.
    """
    pieces: list[str] = []
    current: list[str] = []
    for word in sentence.split():
        candidate = " ".join([*current, word])
        if current and counter(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(
    text: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_sentences: int = 1,
    token_counter: TokenCounter | None = None,
) -> list[Chunk]:
    """Metni cümle sınırlarından, token bütçesine sığan parçalara böler.

    Garantiler (hepsi testlerle doğrulanır):
        - Hiçbir parça ``max_tokens``'ı aşmaz.
        - Hiçbir cümle kaybolmaz; her cümle en az bir parçada yer alır.
        - Parçalar belge sırasını korur.

    Args:
        text: Temizlenmiş metin.
        max_tokens: Parça başına sert token sınırı.
        overlap_sentences: Bir önceki parçadan devralınacak cümle sayısı.
            Bütçeye sığmıyorsa örtüşmeden vazgeçilir.
        token_counter: Modelin gerçek tokenizer'ı. Verilmezse
            :func:`heuristic_token_count` kullanılır.

    Raises:
        ValueError: ``max_tokens`` pozitif değilse veya ``overlap_sentences``
            negatifse.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens pozitif olmalı")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences negatif olamaz")

    counter = token_counter or heuristic_token_count

    sentences: list[str] = []
    for sentence in split_sentences(text):
        if counter(sentence) > max_tokens:
            sentences.extend(_split_oversized(sentence, max_tokens, counter))
        else:
            sentences.append(sentence)
    if not sentences:
        return []

    # (cümleler, ilk cümle sırası) çiftleri
    groups: list[tuple[list[str], int]] = []
    current: list[str] = [sentences[0]]
    start = 0

    for i, sentence in enumerate(sentences[1:], start=1):
        if counter(" ".join([*current, sentence])) <= max_tokens:
            current.append(sentence)
            continue

        groups.append((current, start))
        keep = current[-overlap_sentences:] if overlap_sentences else []
        # Örtüşme bir kolaylık, bütçe ise sert sınır. Devralınan cümleler
        # yeni cümleyle birlikte sığmıyorsa örtüşmeden vazgeçilir — aksi
        # halde parça daha doğarken sınırı aşar.
        while keep and counter(" ".join([*keep, sentence])) > max_tokens:
            keep = keep[1:]
        current = [*keep, sentence]
        start = i - len(keep)

    groups.append((current, start))
    # Not: Burada eskiden "küçük son parçayı öncekiyle birleştir" adımı vardı.
    # Açgözlü paketlemede son parçanın ilk cümlesi, öncekine sığmadığı için
    # oradadır; sert bütçeyle birleştirme hiçbir zaman başarılı olamaz. Adım
    # yalnızca %15 tolerans sayesinde — yani bütçeyi aşarak — çalışıyordu.
    # Tolerans kalkınca ölü kod olduğu için silindi.

    chunks: list[Chunk] = []
    for index, (group, first) in enumerate(groups):
        body = " ".join(group)
        chunks.append(
            Chunk(
                index=index,
                text=body,
                token_count=counter(body),
                first_sentence=first,
                last_sentence=first + len(group) - 1,
            )
        )
    return chunks
