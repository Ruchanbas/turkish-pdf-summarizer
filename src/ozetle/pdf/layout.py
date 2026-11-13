"""Sayfa düzeninden gelen, içerik olmayan metni ayıklar.

İki iş yapar:

1. **Sayfa süslemesi** (header, footer, sayfa numarası): her sayfada tekrar
   eden kısa kenar satırları. Özete girdiklerinde modelin dikkatini dağıtır
   ve "Türk Fizyoterapi Dergisi 2024" cümlesi özetin yarısını kaplayabilir.
2. **Kaynakça**: özetlenecek bir şey içermez ama token bütçesinin önemli bir
   kısmını yer. Belgenin sonunda başlığıyla birlikte kesilir.

Her iki işin de güvenlik kuralları vardır, çünkü yanlış silme geri
getirilemez: uzun satır asla süsleme sayılmaz, kaynakça kesimi metnin
çoğunu götürecekse iptal edilir.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

#: Her sayfanın baş ve sonundan en fazla kaç satır incelenecek.
DEFAULT_EDGE_LINES = 3

#: Bundan uzun satırlar header/footer sayılmaz. Süsleme kısadır; uzun bir
#: satır neredeyse her zaman içeriktir ve silinmesi bilgi kaybıdır.
MAX_FURNITURE_LENGTH = 90

#: Bir kenar satırının süsleme sayılması için görüneceği sayfa oranı.
DEFAULT_MIN_REPEAT_RATIO = 0.5

#: Tekrar tespiti için gereken asgari sayfa sayısı. İki sayfada tekrar eden
#: bir satır tesadüf olabilir.
MIN_PAGES_FOR_REPEAT = 3

#: Kaynakça kesimi metnin bu oranından fazlasını götürecekse iptal edilir.
MAX_REFERENCE_FRACTION = 0.5

_PAGE_NUMBER = re.compile(
    r"""^(
        \d{1,4}                                      # 12
      | [-\u2013\u2014]\s*\d{1,4}\s*[-\u2013\u2014]      # - 12 -
      | \d{1,4}\s*/\s*\d{1,4}                        # 12 / 30
      | (sayfa|page|s\.)\s*\d{1,4}(\s*(/|of|\'den|\'dan)\s*\d{1,4})?
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

_REFERENCE_HEADING = re.compile(
    r"""^\s*
    (?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?             # "7." veya "7.1 "
    (kaynaklar|kaynakça|kaynak\s+listesi|yararlanılan\s+kaynaklar
    |referanslar|references|bibliography|bibliyografya)
    \s*:?\s*$""",
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)

_DIGITS = re.compile(r"\d+")

#: Tek başına duran yıl ("2024"). Başlık sayfalarında sık görülür ve sayfa
#: numarasıyla karıştırılmamalıdır. Bedeli: 1900-2099 arası sayfa
#: numaraları (sürekli numaralı dergiler) yalnızca tekrar kuralıyla
#: yakalanabilir — kaybedilen bir yıl satırı ise geri gelmez.
_YEAR = re.compile(r"^(19|20)\d{2}$")


# ---------------------------------------------------------------------------
# Sayfa süslemesi
# ---------------------------------------------------------------------------


def _normalize_for_compare(line: str) -> str:
    """Sayfadan sayfaya değişen kısımları (sayılar, boşluk) eşitler.

    "Sayfa 3" ile "Sayfa 4" aynı süsleme satırıdır. Tek başına duran yıl
    kendi anahtarını korur; aksi halde her sayfadaki çıplak sayfa
    numarasıyla aynı anahtara düşer ve süsleme sayılır.
    """
    stripped = line.strip()
    if _YEAR.match(stripped):
        return stripped
    return " ".join(_DIGITS.sub("#", stripped).split()).casefold()


def is_page_number(line: str) -> bool:
    """Satır yalnızca bir sayfa numarası mı. Tek başına yıl sayılmaz."""
    stripped = line.strip()
    return bool(_PAGE_NUMBER.match(stripped)) and not _YEAR.match(stripped)


def _edge_indices(lines: Sequence[str], edge_lines: int) -> set[int]:
    """Sayfanın üst ve alt kenarındaki kısa satır bantlarının indeksleri.

    Sabit bir pozisyon penceresi (ilk/son N satır) kısa sayfalarda tüm
    içeriği "kenar" ilan eder. Bunun yerine kenardan içeri doğru, bitişik
    kısa satırlar alınır; ilk uzun satırda bant biter \u2014 gövde başlamıştır.
    """
    indices: set[int] = set()
    for order in (range(len(lines)), range(len(lines) - 1, -1, -1)):
        taken = 0
        for i in order:
            stripped = lines[i].strip()
            if not stripped:
                continue
            if taken >= edge_lines or len(stripped) > MAX_FURNITURE_LENGTH:
                break  # Gövde başladı.
            indices.add(i)
            taken += 1
    return indices


def find_repeating_lines(
    pages: Sequence[str],
    *,
    edge_lines: int = DEFAULT_EDGE_LINES,
    min_ratio: float = DEFAULT_MIN_REPEAT_RATIO,
) -> set[str]:
    """Sayfaların kenarlarında tekrar eden satırları bulur.

    Returns:
        Süsleme sayılan satırların karşılaştırma anahtarları
        (bkz. :func:`_normalize_for_compare`).
    """
    if len(pages) < MIN_PAGES_FOR_REPEAT:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = page.splitlines()
        # set(): aynı sayfada iki kez geçen satır bir kez sayılsın.
        counts.update({_normalize_for_compare(lines[i]) for i in _edge_indices(lines, edge_lines)})

    threshold = max(MIN_PAGES_FOR_REPEAT, min_ratio * len(pages))
    return {key for key, n in counts.items() if n >= threshold}


def strip_page_furniture(
    pages: Sequence[str],
    *,
    edge_lines: int = DEFAULT_EDGE_LINES,
    min_ratio: float = DEFAULT_MIN_REPEAT_RATIO,
) -> list[str]:
    """Her sayfadan header, footer ve sayfa numaralarını siler.

    Sayfalar birleştirilmeden **önce** çağrılmalıdır: birleştikten sonra
    hangi satırın sayfa kenarında olduğu bilgisi kaybolur.

    Yalnızca kenar bandındaki satırlar silinir. Gövdenin ortasında geçen
    aynı metin (örneğin dergi adının atıfta geçmesi) dokunulmadan kalır.
    """
    repeating = find_repeating_lines(pages, edge_lines=edge_lines, min_ratio=min_ratio)
    cleaned: list[str] = []
    for page in pages:
        lines = page.splitlines()
        edges = _edge_indices(lines, edge_lines)
        kept = [
            line
            for i, line in enumerate(lines)
            if not (
                i in edges and (_normalize_for_compare(line) in repeating or is_page_number(line))
            )
        ]
        cleaned.append("\n".join(kept))
    return cleaned


# ---------------------------------------------------------------------------
# Kaynakça
# ---------------------------------------------------------------------------


def cut_references(text: str, *, max_fraction: float = MAX_REFERENCE_FRACTION) -> str:
    """Belge sonundaki kaynakça bölümünü başlığıyla birlikte keser.

    Kurallar:
        - Başlık kendi satırında durmalı. "Kaynaklar gösteriyor ki..." diye
          başlayan bir gövde cümlesi kesim tetiklemez.
        - Son eşleşen başlık kullanılır; içindekiler tablosundaki "Kaynakça"
          satırı belgenin başını kesmez.
        - Güvenlik freni: kesim metnin ``max_fraction``'ından fazlasını
          götürecekse hiçbir şey kesilmez. Yanlış tespitin bedeli belgenin
          tamamı olabilir.
    """
    matches = list(_REFERENCE_HEADING.finditer(text))
    if not matches:
        return text

    cut_at = matches[-1].start()
    removed = len(text) - cut_at
    if removed > max_fraction * len(text):
        return text
    return text[:cut_at].rstrip()
