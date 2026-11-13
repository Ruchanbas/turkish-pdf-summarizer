"""Model katmanı testleri — sahte modellerle, ağ ve GPU gerektirmeden.

Gerçek transformers kodu ``test_models_tiny.py``'de, yerelde sıfırdan
oluşturulan küçük bir modelle; gerçek Hugging Face modelleri ise
``test_models_real.py``'de (``-m model``) test edilir.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import pytest

from ozetle.models.keywords import (
    TURKISH_STOPWORDS,
    KeyBERTExtractor,
    Keyword,
    dedupe_keywords,
)
from ozetle.models.loader import LazyModel, ModelLoadError, check_loading_info
from ozetle.models.summarizer import (
    DEFAULT_MODEL,
    MODELS,
    HFSummarizer,
    ModelSpec,
    get_spec,
    summarize_document,
)
from ozetle.text.clean import turkish_lower


class FakeSummarizer:
    """Her metnin ilk ``keep`` kelimesini 'özet' olarak döndürür."""

    def __init__(self, *, max_input_tokens: int = 50, keep: int = 6) -> None:
        self.max_input_tokens = max_input_tokens
        self.keep = keep
        self.calls: list[list[str]] = []
        self.name = "fake"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def summarize_batch(self, texts: Sequence[str]) -> list[str]:
        self.calls.append(list(texts))
        return [" ".join(t.split()[: self.keep]) + "." for t in texts]


# ---------------------------------------------------------------------------
# LazyModel
# ---------------------------------------------------------------------------


class TestLazyModel:
    def test_ilk_erisime_kadar_yuklenmez(self):
        calls: list[int] = []
        model = LazyModel(lambda: calls.append(1) or "m", name="x")
        assert not calls and not model.loaded
        assert model.get() == "m"
        assert calls == [1] and model.loaded
        assert model.load_seconds is not None

    def test_es_zamanli_erisimde_tek_yukleme(self):
        """Lambda'da ilk istekte birden çok iş parçacığı aynı anda gelebilir."""
        calls: list[int] = []

        def yavas():
            calls.append(1)
            time.sleep(0.05)
            return object()

        model = LazyModel(yavas, name="yavas")
        results: list[object] = []
        threads = [threading.Thread(target=lambda: results.append(model.get())) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1
        assert len({id(r) for r in results}) == 1

    def test_hata_onbellege_alinmaz(self):
        attempts: list[int] = []

        def bazen_basarisiz():
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("ağ hatası")
            return "m"

        model = LazyModel(bazen_basarisiz, name="ag")
        with pytest.raises(ModelLoadError, match="ağ hatası"):
            model.get()
        assert not model.loaded
        assert model.get() == "m"

    def test_model_load_error_sarmalanmaz(self):
        def bozuk():
            raise ModelLoadError("eksik ağırlık")

        with pytest.raises(ModelLoadError, match=r"^eksik ağırlık$"):
            LazyModel(bozuk, name="b").get()

    def test_reset(self):
        calls: list[int] = []
        model = LazyModel(lambda: calls.append(1) or "m", name="x")
        model.get()
        model.reset()
        assert not model.loaded and model.load_seconds is None
        model.get()
        assert len(calls) == 2


class TestCheckLoadingInfo:
    def test_eksik_agirlik_hatadir(self):
        """strict=False felaketinin regresyon testi."""
        info = {"missing_keys": ["decoder.layer.0.weight", "lm_head.weight"]}
        with pytest.raises(ModelLoadError, match="2 ağırlık"):
            check_loading_info(info, model_id="m")

    def test_bagli_agirlik_eksik_sayilmaz(self):
        info = {"missing_keys": ["lm_head.weight"]}
        check_loading_info(info, model_id="m", tied_keys=["lm_head.weight"])

    def test_temiz_yukleme(self):
        check_loading_info({"missing_keys": [], "unexpected_keys": []}, model_id="m")

    def test_beklenmeyen_agirlik_uyaridir(self, caplog):
        check_loading_info({"missing_keys": [], "unexpected_keys": ["pooler.w"]}, model_id="m")
        assert "beklenmeyen" in caplog.text

    def test_uzun_liste_kisaltilir(self):
        info = {"missing_keys": [f"k{i}" for i in range(20)]}
        with pytest.raises(ModelLoadError, match=r"\.\.\."):
            check_loading_info(info, model_id="m")


# ---------------------------------------------------------------------------
# Katalog ve HFSummarizer (sahte tokenizer/model ile)
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_varsayilan_katalogda(self):
        assert DEFAULT_MODEL in MODELS

    def test_anahtar_veya_kimlik(self):
        assert get_spec("mt5-small").model_id == MODELS["mt5-small"].model_id
        assert get_spec(MODELS["mt5-base"].model_id).key == "mt5-base"
        assert get_spec("org/bilinmeyen").model_id == "org/bilinmeyen"


class _FakeEncoding(dict):  # type: ignore[type-arg]
    pass


class _FakeTokenizer:
    def __init__(self) -> None:
        self.seen: list[object] = []

    def __call__(self, text, **kwargs):
        self.seen.append((text, kwargs))
        if isinstance(text, str):
            return {"input_ids": [*text.split(), "</s>"]}
        return _FakeEncoding(input_ids=[t.split() for t in text], token_type_ids=[0])

    def batch_decode(self, generated, skip_special_tokens):
        assert skip_special_tokens
        return [f"  özet   {g} " for g in generated]


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.kwargs.append(kwargs)
        return [len(ids) for ids in kwargs["input_ids"]]  # type: ignore[union-attr]


class TestHFSummarizer:
    def _make(self, **spec_kw):
        tok, model = _FakeTokenizer(), _FakeModel()
        loads: list[str] = []
        spec = ModelSpec(key="t", model_id="org/t", **spec_kw)

        def loader(model_id):
            loads.append(model_id)
            return tok, model

        return HFSummarizer(spec, loader=loader, batch_size=2), tok, model, loads

    def test_tembel_yukleme(self):
        s, _, _, loads = self._make()
        assert not s.loaded and loads == []
        s.warm_up()
        assert s.loaded and loads == ["org/t"]
        assert s.load_seconds is not None

    def test_token_sayimi_onek_dahil(self):
        s, _, _, _ = self._make(prefix="[S2S] ")
        assert s.count_tokens("bir iki") == 4  # "[S2S]" + 2 kelime + </s>

    def test_toplu_ozet_ve_parametreler(self):
        s, tok, model, _ = self._make(max_new_tokens=64, num_beams=2)
        out = s.summarize_batch(["a b", "c d e", "f"])
        assert out == ["özet 2", "özet 3", "özet 1"]  # boşluklar normalize
        assert len(model.kwargs) == 2  # batch_size=2 → iki çağrı
        assert model.kwargs[0]["max_new_tokens"] == 64
        assert model.kwargs[0]["num_beams"] == 2
        assert "token_type_ids" not in model.kwargs[0]
        _, kwargs = tok.seen[0]
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == 512

    def test_ad_ve_pencere(self):
        s, _, _, _ = self._make(max_input_tokens=256)
        assert s.name == "org/t"
        assert s.max_input_tokens == 256


# ---------------------------------------------------------------------------
# Map-reduce
# ---------------------------------------------------------------------------


def _parca(i: int, n: int = 30) -> str:
    return f"Parça {i} " + " ".join(f"k{i}_{j}" for j in range(n)) + "."


class TestSummarizeDocument:
    def test_bos_girdi(self):
        with pytest.raises(ValueError):
            summarize_document([], FakeSummarizer())

    def test_tek_parca_tek_tur(self):
        fake = FakeSummarizer()
        result = summarize_document([_parca(0)], fake)
        assert result.rounds == 1
        assert result.summary == result.partials[0]
        assert len(fake.calls) == 1
        assert result.model == "fake"

    def test_kisa_toplamda_reduce_yapilmaz(self):
        """Regresyon (gerçek modelde görüldü): iki iyi bölüm notu yeniden
        özetlenince model birini atıp diğerini çarpıtarak yeniden yazdı.
        Notlar zaten özet uzunluğundaysa ek model turu yapılmaz."""
        fake = FakeSummarizer(max_input_tokens=100, keep=6)
        result = summarize_document([_parca(i) for i in range(3)], fake)
        assert result.rounds == 1
        assert len(fake.calls) == 1
        assert result.summary == " ".join(result.partials)  # belge sırasıyla

    def test_uzun_toplamda_tek_reduce(self):
        fake = FakeSummarizer(max_input_tokens=100, keep=6)
        result = summarize_document([_parca(i) for i in range(3)], fake, target_words=10)
        assert result.rounds == 2
        assert len(result.partials) == 3
        assert len(fake.calls[1]) == 1  # birleşim tek seferde özetlendi
        assert not result.warnings

    def test_buyuk_birlesim_yeniden_parcalanir(self):
        # 40 parça x 7 kelime = 280 kelime > pencere (50) → reduce'ta yeniden parçalama
        fake = FakeSummarizer(max_input_tokens=50, keep=6)
        result = summarize_document([_parca(i) for i in range(40)], fake, target_words=10)
        assert result.rounds >= 3
        assert len(fake.calls[1]) > 1
        assert all(fake.count_tokens(t) <= 50 for call in fake.calls[1:] for t in call)
        assert result.summary

    def test_tur_siniri(self):
        fake = FakeSummarizer(max_input_tokens=50, keep=6)
        result = summarize_document(
            [_parca(i) for i in range(40)], fake, max_rounds=2, target_words=10
        )
        assert result.rounds == 2
        assert any("2 turda" in w for w in result.warnings)

    def test_bos_ozet_uyarisi(self):
        class Bos(FakeSummarizer):
            def summarize_batch(self, texts):
                return ["" for _ in texts]

        result = summarize_document([_parca(0), _parca(1)], Bos())
        assert result.summary == ""
        assert any("boş özet" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Anahtar kelimeler
# ---------------------------------------------------------------------------


class TestTurkishLower:
    def test_noktali_ve_noktasiz_i(self):
        assert turkish_lower("İÇİN") == "için"
        assert turkish_lower("ILIK") == "ılık"
        assert "İçin".lower() != "için"  # neden gerekli olduğunun kanıtı


class TestKeywords:
    def test_stopword_listesi_turkce_kucuk_harf(self):
        assert all(w == turkish_lower(w) for w in TURKISH_STOPWORDS)
        assert {"ve", "için", "bir", "olarak"} <= TURKISH_STOPWORDS

    def test_tekrar_ayiklama(self):
        pairs = [("İstanbul", 0.9), ("istanbul", 0.8), ("  ", 0.7), ("rehabilitasyon", 0.6)]
        out = dedupe_keywords(pairs, 5)
        assert [k.phrase for k in out] == ["İstanbul", "rehabilitasyon"]

    def test_top_n_siniri(self):
        out = dedupe_keywords([(f"k{i}", 1.0 - i / 10) for i in range(10)], 3)
        assert len(out) == 3

    def test_keybert_mmr_ve_turkce_vektorlestirici(self):
        class FakeKeyBERT:
            def __init__(self):
                self.kwargs = {}

            def extract_keywords(self, text, **kwargs):
                self.kwargs = kwargs
                return [("Fizyoterapi", 0.71234567), ("fizyoterapi", 0.7), ("diz eklemi", 0.6)]

        fake = FakeKeyBERT()
        sentinel = object()
        extractor = KeyBERTExtractor(
            "m", loader=lambda name: fake, vectorizer_factory=lambda: sentinel, diversity=0.4
        )
        assert not extractor.loaded
        out = extractor.extract("Diz eklemi fizyoterapi çalışması.", top_n=5)
        assert extractor.loaded
        assert out == [Keyword("Fizyoterapi", 0.7123), Keyword("diz eklemi", 0.6)]
        assert fake.kwargs["use_mmr"] is True
        assert fake.kwargs["diversity"] == 0.4
        assert fake.kwargs["vectorizer"] is sentinel

    def test_bos_metin_modeli_yuklemez(self):
        extractor = KeyBERTExtractor("m", loader=lambda name: pytest.fail("yüklenmemeli"))
        assert extractor.extract("   ") == []

    def test_gercek_vektorlestirici_turkce_kucultur(self):
        pytest.importorskip("sklearn")
        from ozetle.models.keywords import _turkish_vectorizer

        vec = _turkish_vectorizer()
        vec.fit(["İÇİN bu çalışma İstanbul'da ve Ankara'da 2024 yılında yapıldı"])
        vocab = set(vec.get_feature_names_out())
        assert "istanbul" in vocab
        assert "için" not in vocab and "ve" not in vocab  # stopword'ler gerçekten çalışıyor
        assert not any(ch.isdigit() for term in vocab for ch in term)
        assert not any("̇" in term for term in vocab)


class TestLeadSummarizer:
    def test_ilk_cumleler(self):
        from ozetle.models.summarizer import LeadSummarizer

        s = LeadSummarizer(sentences=2)
        out = s.summarize_batch(["Bir. İki. Üç.", "Tek cümle"])
        assert out == ["Bir. İki.", "Tek cümle"]
        assert s.count_tokens("bir iki") >= 2

    def test_map_reduce_ile(self):
        from ozetle.models.summarizer import LeadSummarizer

        result = summarize_document([_parca(i) for i in range(5)], LeadSummarizer())
        assert result.model == "lead-3"
        assert result.summary.startswith("Parça 0")


class TestTiedToLoaded:
    class _T:
        def __init__(self, ptr: int) -> None:
            self.ptr = ptr

        def data_ptr(self) -> int:
            return self.ptr

    def test_bellek_paylasan_eksik_bagli_sayilir(self):
        from ozetle.models.loader import tied_to_loaded

        state = {"shared.weight": self._T(1), "lm_head.weight": self._T(1), "q.weight": self._T(2)}
        assert tied_to_loaded(state, {"lm_head.weight"}) == {"lm_head.weight"}

    def test_paylasmayan_eksik_gercek_eksiktir(self):
        from ozetle.models.loader import tied_to_loaded

        state = {"shared.weight": self._T(1), "lm_head.weight": self._T(9)}
        assert tied_to_loaded(state, {"lm_head.weight", "yok.weight"}) == set()

    def test_paylasimsiz_model_yeniden_baglanmaz(self):
        from ozetle.models.loader import retie_shared_encoder_decoder

        class Model:
            class config:  # noqa: N801
                tie_encoder_decoder = False

        assert retie_shared_encoder_decoder(Model(), {"decoder.x"}) == set()


class TestDownload:
    def test_safetensors_varsa_bin_indirilmez(self, monkeypatch):
        hub = pytest.importorskip("huggingface_hub")
        from ozetle.models import download

        calls = {}

        class FakeApi:
            def list_repo_files(self, repo_id):
                return ["config.json", "model.safetensors", "pytorch_model.bin", "tf_model.h5"]

        def fake_snapshot(repo_id, ignore_patterns):
            calls["ignore"] = ignore_patterns
            return "/tmp/x"

        monkeypatch.setattr(hub, "HfApi", FakeApi)
        monkeypatch.setattr(hub, "snapshot_download", fake_snapshot)
        assert download.fetch("org/m") == "/tmp/x"
        assert "*.bin" in calls["ignore"] and "*.h5" in calls["ignore"]

    def test_yalniz_bin_varsa_indirilir(self, monkeypatch):
        hub = pytest.importorskip("huggingface_hub")
        from ozetle.models import download

        class FakeApi:
            def list_repo_files(self, repo_id):
                return ["config.json", "pytorch_model.bin"]

        seen = {}
        monkeypatch.setattr(hub, "HfApi", FakeApi)
        monkeypatch.setattr(
            hub,
            "snapshot_download",
            lambda repo_id, ignore_patterns: seen.update(i=ignore_patterns),
        )
        download.fetch("org/eski")
        assert "*.bin" not in seen["i"]

    def test_lead_modu_indirme_yapmaz(self, monkeypatch, capsys):
        from ozetle.models import download

        monkeypatch.setattr(download, "fetch", lambda repo: pytest.fail("indirmemeli"))
        download.main(["--model", "lead", "--no-keywords"])
        assert capsys.readouterr().out == ""


class TestCandidatePhrases:
    @pytest.mark.parametrize(
        "fiil",
        ["gözlendi", "ölçüldü", "verilmiştir", "artmaktadır", "azalttı", "yapıldı", "bulundu",
         "önemlidir", "artıyor", "gösterdi", "edildi"],
    )  # fmt: skip
    def test_cekimli_fiil_elenir(self, fiil):
        from ozetle.models.keywords import candidate_phrases

        assert candidate_phrases(fiil) == []

    @pytest.mark.parametrize(
        "isim",
        ["kuvveti", "fizyoterapi", "rehabilitasyon", "kültür", "beklenti", "osteoartriti",
         "hastaların", "fonksiyonel", "tedavi", "programı"],
    )  # fmt: skip
    def test_ti_di_ile_biten_isim_korunur(self, isim):
        """'kas kuvveti' gibi alanın en önemli terimleri fiil sanılmamalı."""
        from ozetle.models.keywords import candidate_phrases

        assert candidate_phrases(isim) == [isim]

    def test_kisaltma_ve_sayi_elenir(self):
        from ozetle.models.keywords import candidate_phrases

        assert candidate_phrases("500 mg doz, Dr. Yılmaz") == ["doz", "yılmaz"]

    def test_ikili_cumle_sinirini_asmaz(self):
        """Regresyon (gerçek modelde görüldü): 'sonuçları özetlemektedir.
        Ortalama' → fiil elenince 'sonuçları ortalama' ikilisi doğuyordu."""
        from ozetle.models.keywords import candidate_phrases

        out = candidate_phrases("Tablo bu sonuçları özetlemektedir. Ortalama değer yüksek")
        assert "sonuçları ortalama" not in out
        assert "ortalama değer" in out

    def test_ikili_elenen_kelimeyi_atlamaz(self):
        from ozetle.models.keywords import candidate_phrases

        assert "iyileşme fizyoterapist" not in candidate_phrases("iyileşme gözlendi fizyoterapist")


class TestPairFilter:
    @pytest.mark.parametrize(
        "cop",
        ["egzersizi tedavinin", "egzersiz grubunda", "osteoartritinde ağrıyı",
         "yetişkinlerde ağrının", "etkinin kas", "hastalıkta ağrı", "dakikada nabız"],
    )  # fmt: skip
    def test_hal_ekli_ikili_kurulmaz(self, cop):
        """Gerçek modelde görüldü: iki ayrı öbeğin sınırını aşan ikililer."""
        from ozetle.models.keywords import candidate_phrases

        assert not [p for p in candidate_phrases(cop) if " " in p]

    @pytest.mark.parametrize(
        "iyi",
        ["kas kuvveti", "diz osteoartriti", "egzersiz programı", "vitamin eksikliği",
         "insülin direnci", "beden eğitimi", "fiziksel aktivite", "yaşam kalitesi",
         "fonksiyonel kapasite", "makine öğrenmesi"],
    )  # fmt: skip
    def test_gecerli_ikili_korunur(self, iyi):
        """'aktivite' ünlü + -te ile biter ama bulunma eki değildir (ses uyumu)."""
        from ozetle.models.keywords import candidate_phrases

        assert iyi in candidate_phrases(iyi)


class TestKeywordRound2:
    """Ekran görüntüsü koşusunda gerçek modelle görülen adaylar."""

    @pytest.mark.parametrize(
        "olmamali",
        ["hangi egzersiz", "ağrı arasındaki", "kuvvetindeki değişimle", "yürüme kapasitesini",
         "yetişkinlerde", "ağrıyı"],
    )  # fmt: skip
    def test_elenir(self, olmamali):
        from ozetle.models.keywords import candidate_phrases

        assert olmamali not in candidate_phrases(olmamali)

    @pytest.mark.parametrize("iyi", ["makale yazımı", "klinik kılavuzlar", "yürüme kapasitesi"])
    def test_korunur(self, iyi):
        from ozetle.models.keywords import candidate_phrases

        assert iyi in candidate_phrases(iyi)
