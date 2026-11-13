"""Anahtar kelime çıkarma: KeyBERT, MMR ve gerçekten çalışan Türkçe stopword'ler.

Eski kodda üç sessiz hata vardı:

1. ``stop_words="turkish"`` geçiliyordu. scikit-learn'ün yerleşik listesi
   yalnızca İngilizcedir; "turkish" değeri ya hata verir ya da hiçbir şey
   yapmaz. Sonuç: "bir", "ve", "için" anahtar kelime çıkıyordu.
2. MMR kapalıydı; ilk beş sonuç aynı kökün beş çekimiydi ("hasta",
   "hastalar", "hastaların"...). MMR alaka ile çeşitliliği dengeler.
3. KeyBERT'in vektörleştiricisi ``str.lower()`` kullanır. "İçin" → "i̇çin"
   (birleşik nokta) olur ve stopword listesindeki "için" ile eşleşmez.
   Burada Türkçe küçültme ön işlemcisi verilir.

Gerçek modelle ilk denemede bir sorun daha görüldü: "ölçüldü dr", "toplam mg"
gibi adaylar. Kısaltmalar (iki harf) ve çekimli fiiller ("gözlendi",
"verilmiştir") anahtar kelime olamaz; :func:`candidate_phrases` bunları eler
ve metinde gerçekten yan yana durmayan kelimelerden ikili kurmaz.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol

from ozetle.models.loader import LazyModel
from ozetle.text.clean import turkish_lower

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# fmt: off
TURKISH_STOPWORDS: frozenset[str] = frozenset({
    "acaba", "ama", "ancak", "artık", "aslında", "az", "bana", "bazen", "bazı", "bazıları",
    "belki", "ben", "beni", "benim", "beri", "bile", "bir", "biraz", "birçok", "biri",
    "birkaç", "birlikte", "birşey", "biz", "bize", "bizi", "bizim", "bu", "buna", "bunda",
    "bundan", "bunlar", "bunları", "bunların", "bunu", "bunun", "burada", "böyle", "bütün",
    "çok", "çünkü", "da", "daha", "dahi", "de", "değil", "diğer", "diye", "dolayı", "en",
    "edilen", "edildi", "edilmiştir", "eğer", "elbette", "gibi", "göre", "hala", "halde",
    "hem", "hep", "hepsi", "her", "herhangi", "hiç", "için", "ile", "ilgili", "ise", "işte",
    "kadar", "karşı", "kendi", "kendine", "kez", "ki", "kim", "kimse", "mi", "mı", "mu", "mü",
    "nasıl", "ne", "neden", "nedeniyle", "nerede", "niçin", "o", "olan", "olarak", "oldu",
    "olduğu", "olduğunu", "olmak", "olması", "olmayan", "olmuştur", "olup", "on", "ona",
    "onlar", "onları", "onların", "onu", "onun", "orada", "öyle", "önce", "sadece", "sanki",
    "sonra", "şey", "şimdi", "şu", "şuna", "şunu", "tarafından", "tüm", "üzere", "var",
    "vardır", "ve", "veya", "ya", "yani", "yapılan", "yapılmıştır", "yine", "yok", "zaten",
    "aynı", "arasında", "ayrıca", "bulunan", "fakat", "iki", "ilk", "son", "yeni", "üç",
})
# fmt: on


_WORD = re.compile(r"[^\W\d_]{3,}")

#: Çekimli fiil sonları. Bilinçli olarak dar tutuldu: Türkçede isimler de
#: -ti/-di ile bitebilir ("kas kuvveti", "beklenti", "kültür"); bunlar
#: elenirse en değerli anahtar kelimeler kaybolur. Yalnızca fiile özgü
#: biçimler yakalanır:
#:   yapılmış(tır) · artmaktadır · artıyor · ölçüldü / bulundu / verildi ·
#:   gözlendi · gösterdi · azalttı · önemlidir
_VERB_FORM = re.compile(
    r"(?:m[ıiuü]ş(?:t[ıiuü]r)?|m[ae]kt[ae]d[ıi]r|[ıiuü]yor(?:du|dı|lar)?"
    r"|[ıiuü][lnr]d[ıiuü]|l[ae]nd[ıi]|[ae]rd[ıi]|tt[ıiuü]|d[ıiuü]r)$"
)


#: Cümle ve öbek sınırları: ikili ifadeler bunları aşamaz.
_BOUNDARY = re.compile(r"[.!?;:,()\[\]\"«»/]+")
_ANY_WORD = re.compile(r"[^\W_]+")


def _keep(word: str) -> bool:
    if len(word) < 3 or not word.isalpha() or word in TURKISH_STOPWORDS:
        return False
    return not (len(word) > 5 and _VERB_FORM.search(word))


def candidate_phrases(text: str) -> list[str]:
    """Anahtar kelime adayları: tek kelimeler ve **gerçekten yan yana** ikililer.

    scikit-learn'ün n-gram üreticisi stopword'leri attıktan sonra kalan
    kelimeleri birleştirir; aradaki fiil elenince cümle sınırının iki yanı
    yan yana düşer ve "sonuçları ortalama" gibi metinde hiç geçmeyen ikililer
    doğar. Burada ikili yalnızca aynı cümle öbeğinde, arada elenen kelime
    olmadan yan yana duran iki aday kelimeden kurulur.

    >>> candidate_phrases("Kas kuvveti ölçüldü. Sonuçlar gözlendi; dr mg")
    ['kas', 'kuvveti', 'kas kuvveti', 'sonuçlar']
    """
    out: list[str] = []
    for segment in _BOUNDARY.split(turkish_lower(text)):
        run: list[str] = []
        for word in [*_ANY_WORD.findall(segment), ""]:
            if word and _keep(word):
                run.append(word)
                continue
            out.extend(run)
            out.extend(f"{a} {b}" for a, b in pairwise(run))
            run = []
    return out


@dataclass(frozen=True, slots=True)
class Keyword:
    phrase: str
    score: float


class KeywordExtractor(Protocol):
    def extract(self, text: str, *, top_n: int = 8) -> list[Keyword]: ...


def _turkish_vectorizer() -> Any:
    """Adayları :func:`candidate_phrases` ile üreten vektörleştirici (scikit-learn)."""
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(analyzer=candidate_phrases, lowercase=False)


def _load_keybert(model_name: str) -> Any:
    from keybert import KeyBERT

    return KeyBERT(model=model_name)


def dedupe_keywords(pairs: Sequence[tuple[str, float]], top_n: int) -> list[Keyword]:
    """Aynı ifadenin büyük/küçük harf varyantlarını ve tekrarları ayıklar.

    >>> [k.phrase for k in dedupe_keywords([("Fizyoterapi", 0.9), ("fizyoterapi", 0.8)], 5)]
    ['Fizyoterapi']
    """
    seen: set[str] = set()
    out: list[Keyword] = []
    for phrase, score in pairs:
        key = turkish_lower(phrase).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(Keyword(phrase=phrase.strip(), score=round(float(score), 4)))
        if len(out) >= top_n:
            break
    return out


class KeyBERTExtractor:
    """Çok dilli cümle gömmeleri ile KeyBERT, MMR açık."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        loader: Callable[[str], Any] = _load_keybert,
        vectorizer_factory: Callable[[], Any] = _turkish_vectorizer,
        diversity: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self._model: LazyModel[Any] = LazyModel(lambda: loader(model_name), name=model_name)
        self._vectorizer_factory = vectorizer_factory
        self._diversity = diversity

    @property
    def loaded(self) -> bool:
        return self._model.loaded

    def warm_up(self) -> None:
        self._model.get()

    def extract(self, text: str, *, top_n: int = 8) -> list[Keyword]:
        if not text.strip():
            return []
        pairs = self._model.get().extract_keywords(
            text,
            vectorizer=self._vectorizer_factory(),
            use_mmr=True,
            diversity=self._diversity,
            top_n=top_n * 2,  # tekrar ayıklamasından sonra top_n kalsın
        )
        return dedupe_keywords(pairs, top_n)
