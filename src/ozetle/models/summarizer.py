"""Özetleme: Hugging Face seq2seq modelleri ve uzun belge için map-reduce.

Model penceresi (512-768 token) bir makalenin küçük bir kısmını taşır. Uzun
belge şöyle özetlenir:

    parçalar ──map──▶ parça özetleri ──(gerekirse) reduce──▶ özet

Reduce **yalnızca gerektiğinde** yapılır: parça özetlerinin toplamı zaten
bir özet uzunluğundaysa (``target_words``) olduğu gibi birleştirilir. Neden:
haber verisiyle eğitilmiş, tek belge için yapılmış bir model, kendi
yazdığı iki notu birleştirip yeniden özetlerken birini atıp diğerini
yeniden yazıyor. Örnek makalede nihai özet yalnızca ikinci bölümden geldi
ve bir sınırlılığı ("katılımcıların tek merkezden seçilmesi") bir öneriye
çevirdi. Her ek tur bilgi kaybı ve uydurma riskidir; kısa belgede gereksiz.

Toplam uzunsa birleşim pencereye sığdığında bir kez daha özetlenir,
sığmadığında yeniden parçalanıp aynı işlem tekrarlanır. ``max_rounds`` bir
üst sınır koyar.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from ozetle.models.loader import (
    LazyModel,
    check_loading_info,
    retie_shared_encoder_decoder,
    tied_to_loaded,
)
from ozetle.text.chunk import chunk_text, heuristic_token_count, split_sentences
from ozetle.text.clean import tidy_output

logger = logging.getLogger(__name__)


class Summarizer(Protocol):
    """Özetleme modelinin arayüzü. Testler sahte bir uygulama kullanır."""

    @property
    def name(self) -> str: ...

    @property
    def max_input_tokens(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    def summarize_batch(self, texts: Sequence[str]) -> list[str]: ...


# ---------------------------------------------------------------------------
# Model kataloğu
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Bir özetleme modelinin çalıştırma ayarları."""

    key: str
    model_id: str
    max_input_tokens: int = 512
    prefix: str = ""
    """Bazı modeller eğitimde bir görev öneki ile beslenmiştir."""
    max_new_tokens: int = 128
    min_new_tokens: int = 0
    """Asgari özet uzunluğu; tek kelimelik "özetleri" engeller."""
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    license_note: str = ""


#: Faz 2'de karşılaştırılan adaylar. Varsayılan model benchmark sonucuna göre
#: seçilir (bkz. README → Benchmark).
MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec("bert2bert", "mrm8488/bert2bert_shared-turkish-summarization"),
        # Model kartı: eğitimde 784 token girdi. mT5 göreli konum kodlaması
        # kullandığı için 512 sınırı yok; daha uzun pencere = daha az parça.
        ModelSpec("mt5-small", "ozcangundes/mt5-small-turkish-summarization", max_input_tokens=768),
        ModelSpec("mt5-base", "mukayese/mt5-base-turkish-summarization"),
        # Model kartı yok. Önek ölçülerek seçildi (bench/run.py prefix, n=8):
        # "" 25.87, "[S2S] " 24.98, "[S2S]" 25.89 ROUGE-L — fark gürültü
        # düzeyinde, en sade olan seçildi.
        ModelSpec(
            "turna",
            "boun-tabi-LMG/turna_summarization_mlsum",
            license_note="TURNA yalnızca ticari olmayan akademik kullanım için paylaşılmıştır.",
        ),
    )
}

DEFAULT_MODEL = "mt5-small"

#: Model gerektirmeyen çıkarımsal taban çizgisi (bkz. :class:`LeadSummarizer`).
LEAD_MODEL = "lead"


def get_spec(key_or_id: str) -> ModelSpec:
    """Katalog anahtarı veya doğrudan Hugging Face kimliği ile model ayarı."""
    if key_or_id in MODELS:
        return MODELS[key_or_id]
    for spec in MODELS.values():
        if spec.model_id == key_or_id:
            return spec
    return ModelSpec(key=key_or_id, model_id=key_or_id)


# ---------------------------------------------------------------------------
# Taban çizgisi: ilk cümleler
# ---------------------------------------------------------------------------


class LeadSummarizer:
    """Her metnin ilk ``sentences`` cümlesini döndüren çıkarımsal özetleyici.

    Haber özetlemede "lead-3" klasik taban çizgisidir; bir model bunu
    geçemiyorsa çalıştırmaya değmez. Ayrıca hiçbir ML bağımlılığı
    gerektirmediği için arayüz geliştirme ve hızlı deneme için kullanılır:
    ``OZETLE_MODEL=lead``.
    """

    name = "lead-3"
    max_input_tokens = 450

    def __init__(self, sentences: int = 3) -> None:
        self._sentences = sentences

    def count_tokens(self, text: str) -> int:
        return heuristic_token_count(text)

    def summarize_batch(self, texts: Sequence[str]) -> list[str]:
        return [" ".join(split_sentences(t)[: self._sentences]) for t in texts]


# ---------------------------------------------------------------------------
# Hugging Face uygulaması
# ---------------------------------------------------------------------------


def _inference_mode() -> contextlib.AbstractContextManager[Any]:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch yoksa zaten model de yok
        return contextlib.nullcontext()
    return cast("contextlib.AbstractContextManager[Any]", torch.inference_mode())


def load_seq2seq(model_id: str) -> tuple[Any, Any]:
    """Tokenizer ve modeli **katı** biçimde yükler.

    Eksik ağırlık varsa :class:`~ozetle.models.loader.ModelLoadError` yükselir.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model, info = AutoModelForSeq2SeqLM.from_pretrained(model_id, output_loading_info=True)
    missing = set(info.get("missing_keys", ()))
    tied = retie_shared_encoder_decoder(model, missing)
    tied |= tied_to_loaded(model.state_dict(), missing - tied)
    check_loading_info(info, model_id=model_id, tied_keys=tied)
    model.eval()
    return tokenizer, model


class HFSummarizer:
    """Hugging Face seq2seq modeli ile özetleyici.

    Tokenizer ve model ayrı ayrı tembel yüklenir: token sayımı (parçalama
    için) yalnızca küçük tokenizer'ı gerektirir, 1 GB'lık model ilk özet
    isteğine kadar belleğe girmez.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        loader: Callable[[str], tuple[Any, Any]] = load_seq2seq,
        batch_size: int = 4,
    ) -> None:
        self.spec = spec
        self._batch_size = batch_size
        self._bundle: LazyModel[tuple[Any, Any]] = LazyModel(
            lambda: loader(spec.model_id), name=spec.model_id
        )

    @property
    def name(self) -> str:
        return self.spec.model_id

    @property
    def max_input_tokens(self) -> int:
        return self.spec.max_input_tokens

    @property
    def loaded(self) -> bool:
        return self._bundle.loaded

    @property
    def load_seconds(self) -> float | None:
        return self._bundle.load_seconds

    def warm_up(self) -> None:
        """Modeli şimdi yükler (Lambda soğuk başlangıcında, istekten önce)."""
        self._bundle.get()

    def count_tokens(self, text: str) -> int:
        tokenizer, _ = self._bundle.get()
        ids = tokenizer(self.spec.prefix + text, add_special_tokens=True)["input_ids"]
        return len(ids)

    def summarize_batch(self, texts: Sequence[str]) -> list[str]:
        tokenizer, model = self._bundle.get()
        outputs: list[str] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [self.spec.prefix + t for t in texts[start : start + self._batch_size]]
            encoded = tokenizer(
                batch,
                max_length=self.spec.max_input_tokens,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            # bert2bert gibi encoder-decoder modeller token_type_ids kabul etmez.
            encoded.pop("token_type_ids", None)
            with _inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=self.spec.max_new_tokens,
                    min_new_tokens=self.spec.min_new_tokens,
                    num_beams=self.spec.num_beams,
                    no_repeat_ngram_size=self.spec.no_repeat_ngram_size,
                    early_stopping=True,
                )
            outputs.extend(
                tidy_output(s) for s in tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return outputs


# ---------------------------------------------------------------------------
# Map-reduce
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    partials: list[str]
    """Her giriş parçasının özeti (map turu). Arayüzde parçayla yan yana gösterilir."""
    rounds: int
    """Toplam model turu. Tek parçalı belge için 1."""
    model: str
    seconds: float
    warnings: list[str] = field(default_factory=list)


def join_notes(notes: Sequence[str]) -> str:
    """Notları cümle olarak birleştirir.

    Model çıktısı ``max_new_tokens`` sınırında veya noktasız bitebilir;
    düz birleştirme "…bir araya getirmiştir Yüksek devam oranı…" gibi iki
    cümleyi tek cümleye yapıştırır. Noktalamasız biten nota nokta eklenir.

    >>> join_notes(["Bir araya getirmiştir", "Devam oranı yüksek."])
    'Bir araya getirmiştir. Devam oranı yüksek.'
    """
    parts = [n.strip() for n in notes if n.strip()]
    return " ".join(p if p[-1] in ".!?…" else p + "." for p in parts)


#: Bir özetin okunabilir uzunluğu (kelime). Parça özetlerinin toplamı bunu
#: aşmıyorsa ek model turu yapılmaz.
DEFAULT_TARGET_WORDS = 120


def summarize_document(
    chunks: Sequence[str],
    summarizer: Summarizer,
    *,
    max_rounds: int = 4,
    budget_margin: int = 16,
    target_words: int = DEFAULT_TARGET_WORDS,
) -> SummaryResult:
    """Parçalardan tek bir özet üretir (map-reduce).

    Args:
        chunks: :func:`ozetle.pipeline.prepare` çıktısındaki parça metinleri.
        summarizer: Özetleme modeli.
        max_rounds: Model turlarının üst sınırı. Aşılırsa son turun özetleri
            birleştirilip döndürülür ve uyarı eklenir.
        budget_margin: Reduce parçalamasında pencereden bırakılan pay
            (önek ve özel tokenlar için).
        target_words: Parça özetlerinin toplamı bundan kısaysa reduce
            yapılmaz; notlar belge sırasıyla birleştirilir.

    Raises:
        ValueError: Parça yoksa.
    """
    if not chunks:
        raise ValueError("Özetlenecek parça yok")

    started = time.perf_counter()
    warnings: list[str] = []
    partials = summarizer.summarize_batch(list(chunks))
    texts = [p for p in partials if p.strip()]
    rounds = 1
    budget = max(32, summarizer.max_input_tokens - budget_margin)

    def words(items: list[str]) -> int:
        return sum(len(t.split()) for t in items)

    while len(texts) > 1 and words(texts) > target_words:
        if rounds >= max_rounds:
            warnings.append(
                f"Özet {max_rounds} turda tek parçaya inmedi; son tur özetleri birleştirildi."
            )
            break
        joined = join_notes(texts)
        if summarizer.count_tokens(joined) <= budget:
            texts = summarizer.summarize_batch([joined])
        else:
            regrouped = chunk_text(
                joined,
                max_tokens=budget,
                overlap_sentences=0,
                token_counter=summarizer.count_tokens,
            )
            texts = summarizer.summarize_batch([c.text for c in regrouped])
        texts = [t for t in texts if t.strip()]
        rounds += 1
        logger.debug("Reduce turu %d: %d özet", rounds, len(texts))

    summary = join_notes(texts)
    if not summary:
        warnings.append("Model boş özet üretti.")
    return SummaryResult(
        summary=summary,
        partials=partials,
        rounds=rounds,
        model=summarizer.name,
        seconds=time.perf_counter() - started,
        warnings=warnings,
    )
