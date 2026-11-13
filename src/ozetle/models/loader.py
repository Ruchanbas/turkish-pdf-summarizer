"""Model yükleme: tembel, iş parçacığı güvenli, sessiz hata yok.

Eski kodun iki sorunu vardı:

1. **İçe aktarma = indirme.** Model modül seviyesinde yükleniyordu; testler,
   CLI'ın ``--help`` çıktısı, hatta bir tip kontrolü bile 1 GB'lık modeli
   indirmeyi tetikliyordu. Burada model ilk gerçekten istendiğinde yüklenir.
2. **``strict=False``.** Ağırlıklar ``load_state_dict(..., strict=False)`` ile
   yükleniyordu. Eksik katmanlar sessizce rastgele başlatılıyor, model hata
   vermek yerine akıcı ama anlamsız Türkçe üretiyordu. Burada eksik ağırlık
   bir hatadır: :func:`check_loading_info`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Generic, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelLoadError(RuntimeError):
    """Model yüklenemedi veya eksik ağırlıkla yüklendi."""


class LazyModel(Generic[T]):
    """İlk erişimde bir kez yüklenen, iş parçacıkları arasında paylaşılan nesne.

    Çift kontrollü kilitleme kullanır: yüklendikten sonra ``get()`` kilit
    almadan döner. Yükleme başarısız olursa sonuç önbelleğe alınmaz; bir
    sonraki çağrı yeniden dener (geçici ağ hatası kalıcı arızaya dönüşmesin).

    >>> calls = []
    >>> m = LazyModel(lambda: calls.append(1) or "model", name="test")
    >>> m.loaded
    False
    >>> m.get(), m.get(), len(calls)
    ('model', 'model', 1)
    """

    def __init__(self, factory: Callable[[], T], *, name: str) -> None:
        self._factory = factory
        self._name = name
        self._lock = threading.Lock()
        self._value: T | None = None
        self._loaded = False
        self.load_seconds: float | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def loaded(self) -> bool:
        return self._loaded

    def get(self) -> T:
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._load()
        return cast(T, self._value)

    def _load(self) -> None:
        started = time.perf_counter()
        logger.info("Model yükleniyor: %s", self._name)
        try:
            value = self._factory()
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"{self._name} yüklenemedi: {exc}") from exc
        self._value = value
        self.load_seconds = time.perf_counter() - started
        self._loaded = True
        logger.info("Model hazır: %s (%.1f sn)", self._name, self.load_seconds)

    def reset(self) -> None:
        """Yüklenmiş nesneyi bırakır. Testler ve bellek baskısı için."""
        with self._lock:
            self._value = None
            self._loaded = False
            self.load_seconds = None


def check_loading_info(
    info: Mapping[str, Iterable[str]],
    *,
    model_id: str,
    tied_keys: Iterable[str] = (),
) -> None:
    """``from_pretrained(..., output_loading_info=True)`` çıktısını denetler.

    Eksik ağırlık varsa :class:`ModelLoadError` yükseltir. Bağlı (tied)
    ağırlıklar, örneğin çıkış katmanının gömme matrisini paylaşması, kayıtta
    ayrıca bulunmaz ve eksik sayılmaz.

    Beklenmeyen (fazla) ağırlıklar yalnızca loglanır: çoğu zaman eğitimden
    kalan, çıkarımda kullanılmayan katmanlardır ve modeli bozmaz.
    """
    tied = set(tied_keys)
    missing = sorted(k for k in info.get("missing_keys", ()) if k not in tied)
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise ModelLoadError(
            f"{model_id}: {len(missing)} ağırlık kayıtta yok ve rastgele başlatılırdı "
            f"({shown}). Yanlış mimari veya bozuk kayıt; model anlamsız çıktı üretir."
        )
    unexpected = list(info.get("unexpected_keys", ()))
    if unexpected:
        logger.warning("%s: %d beklenmeyen ağırlık yok sayıldı.", model_id, len(unexpected))


def retie_shared_encoder_decoder(model: Any, missing: Iterable[str]) -> set[str]:
    """Paylaşımlı encoder-decoder kayıtlarında decoder'ı encoder'a yeniden bağlar.

    ``tie_encoder_decoder=True`` ile eğitilmiş modellerin (ör. bert2bert
    "shared") kaydında decoder'ın encoder ile ortak katmanları yoktur;
    yükleyici onları encoder'a bağlamalıdır. transformers 4 bunu yapıyordu,
    transformers 5 eski biçimdeki kayıtlarda yapmıyor: katmanlar rastgele
    başlatılıyor ve yalnızca bir uyarı yazılıyor. Model hata vermeden
    anlamsız özet üretiyor.

    Yalnızca "eksik" görünen katmanlar değil, encoder'da karşılığı olan
    **bütün** decoder katmanları bağlanır: transformers 5 decoder'ın kelime
    gömmelerini çıkış katmanının kopyasından doldurur, bu yüzden onlar eksik
    görünmez ama yine de encoder'a bağlı değildir. Bağ kopya değil, aynı
    parametre nesnesidir.

    Returns:
        Eksik listesinden bu işlemle bağlanan anahtarlar. Kalanlar gerçek eksiktir.
    """
    if not getattr(getattr(model, "config", None), "tie_encoder_decoder", False):
        return set()
    base_prefix = model.decoder.base_model_prefix
    decoder_base = getattr(model.decoder, base_prefix, None)
    if decoder_base is None:
        return set()

    encoder_params = dict(model.encoder.named_parameters(remove_duplicate=False))
    retied: set[str] = set()
    for name, _ in list(decoder_base.named_parameters(remove_duplicate=False)):
        if name not in encoder_params:
            continue  # çapraz dikkat gibi decoder'a özgü katman
        module_path, _, attr = name.rpartition(".")
        module = decoder_base.get_submodule(module_path) if module_path else decoder_base
        setattr(module, attr, encoder_params[name])
        retied.add(f"decoder.{base_prefix}.{name}")

    # Çıkış katmanı decoder'ın (artık encoder'la ortak) gömme matrisini paylaşır.
    output = model.decoder.get_output_embeddings()
    if output is not None and getattr(model.decoder.config, "tie_word_embeddings", True):
        output.weight = decoder_base.get_input_embeddings().weight

    logger.info("%d decoder ağırlığı encoder'a bağlandı (paylaşımlı model).", len(retied))
    return {k for k in missing if k in retied}


def tied_to_loaded(state_dict: Mapping[str, Any], missing: Iterable[str]) -> set[str]:
    """Eksik görünen ama yüklenmiş bir ağırlıkla aynı belleği paylaşan anahtarlar.

    Bağlı ağırlıklar (çıkış katmanının gömme matrisini paylaşması gibi)
    kayıtta ayrıca bulunmaz. Bağlılık yapılandırmaya güvenilerek değil,
    bellek adresine bakılarak **kanıtlanır**: paylaşmıyorsa gerçekten eksiktir.
    """
    missing = set(missing)
    loaded = {t.data_ptr() for k, t in state_dict.items() if k not in missing}
    return {k for k in missing if k in state_dict and state_dict[k].data_ptr() in loaded}
