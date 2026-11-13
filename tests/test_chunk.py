"""Cümle bölme ve parçalama testleri.

Parçalayıcının üç garantisi var ve her biri burada ayrı ayrı, rastgele
üretilmiş metinler dahil, doğrulanır:

1. Hiçbir parça token bütçesini aşmaz.
2. Hiçbir cümle kaybolmaz.
3. Belge sırası korunur.
"""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from ozetle.text.chunk import (
    Chunk,
    chunk_text,
    heuristic_token_count,
    split_sentences,
)


def word_count(text: str) -> int:
    """Testlerde öngörülebilir sınırlar için: 1 kelime = 1 token."""
    return len(text.split())


class TestSplitSentences:
    def test_basit(self):
        assert split_sentences("Bir. İki! Üç?") == ["Bir.", "İki!", "Üç?"]

    @pytest.mark.parametrize(
        "cumle",
        [
            "Dr. Ayşe Yılmaz ölçümleri yaptı.",
            "Prof. Dr. Mehmet Kaya danışmandı.",
            "Kas, eklem vb. yapılar incelendi.",
            "Demir ve ark. benzer sonuç buldu.",
            "Ortalama değer 3.14 olarak ölçüldü.",
            "Toplam 1.250 hasta katıldı.",
            "Çalışmanın 2. bölümünde ayrıntılar var.",
            "M. Kemal caddesinde yapıldı.",
            "Bkz. Tab. 3 ve Şek. 2 karşılaştırması.",
        ],
    )
    def test_cumle_bitirmeyen_noktalar_bolmez(self, cumle):
        assert split_sentences(f"{cumle} Sonraki cümle.") == [cumle, "Sonraki cümle."]

    def test_turkce_buyuk_harfle_baslayan_cumle(self):
        assert split_sentences("Bitti. İkinci başladı. Şimdi üçüncü. Çok güzel.") == [
            "Bitti.",
            "İkinci başladı.",
            "Şimdi üçüncü.",
            "Çok güzel.",
        ]

    def test_tirnak_icinde_bitis(self):
        assert split_sentences('"Bitti." Sonra başladı.') == ['"Bitti."', "Sonra başladı."]

    def test_paragraf_siniri_cumle_siniridir(self):
        assert split_sentences("Noktasız başlık\n\nGövde cümlesi.") == [
            "Noktasız başlık",
            "Gövde cümlesi.",
        ]

    def test_kisaltmaya_benzeyen_fiil_cumleyi_bitirir(self):
        """'der' kısaltma listesinde olsaydı bu iki cümle birleşirdi."""
        assert split_sentences("Ahmet gel der. Sonra gider.") == ["Ahmet gel der.", "Sonra gider."]

    def test_kucuk_harfle_devam_bolmez(self):
        """Nokta sonrası küçük harf: büyük ihtimalle kısaltma veya kesme."""
        assert split_sentences("Değer yak. olarak ölçüldü.") == ["Değer yak. olarak ölçüldü."]

    @pytest.mark.parametrize("bos", ["", "   ", "\n\n"])
    def test_bos(self, bos):
        assert split_sentences(bos) == []

    def test_sentinel_karakteri_metne_sizmaz(self):
        sonuc = split_sentences("Dr. A geldi.\x00 Bitti.")
        assert all("\x00" not in s for s in sonuc)


class TestHeuristicTokenCount:
    def test_muhafazakar_tahmin(self):
        """Tahmin kelime sayısından yüksek olmalı: az tahmin pencereyi taşırır."""
        metin = "Hastaların fonksiyonel skorlarında anlamlı iyileşme gözlendi."
        assert heuristic_token_count(metin) > len(metin.split())

    def test_bos(self):
        assert heuristic_token_count("") == 0


def _metin(n: int, kelime: int = 10) -> str:
    return " ".join(
        f"Cümle {i} " + " ".join(f"k{i}_{j}" for j in range(kelime - 2)) + "." for i in range(n)
    )


def _cumle_kapsami(chunks: list[Chunk]) -> list[int]:
    seen: list[int] = []
    for c in chunks:
        seen.extend(range(c.first_sentence, c.last_sentence + 1))
    return seen


class TestChunkText:
    def test_bos_metin(self):
        assert chunk_text("") == []

    def test_kisa_metin_tek_parca(self):
        chunks = chunk_text("Kısa bir metin. İki cümle.", max_tokens=100, token_counter=word_count)
        assert len(chunks) == 1
        assert chunks[0].text == "Kısa bir metin. İki cümle."
        assert (chunks[0].first_sentence, chunks[0].last_sentence) == (0, 1)

    @pytest.mark.parametrize("butce", [15, 25, 40, 100])
    def test_butce_asilmaz(self, butce):
        chunks = chunk_text(_metin(60), max_tokens=butce, token_counter=word_count)
        assert chunks
        assert all(c.token_count <= butce for c in chunks)
        assert all(word_count(c.text) <= butce for c in chunks)

    def test_hicbir_cumle_kaybolmaz(self):
        metin = _metin(60)
        n = len(split_sentences(metin))
        chunks = chunk_text(metin, max_tokens=35, token_counter=word_count)
        assert set(_cumle_kapsami(chunks)) == set(range(n))
        for i in range(n):
            assert f"Cümle {i} " in " ".join(c.text for c in chunks)

    def test_sira_korunur(self):
        chunks = chunk_text(_metin(40), max_tokens=30, token_counter=word_count)
        starts = [c.first_sentence for c in chunks]
        assert starts == sorted(starts)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_ortusme_var(self):
        chunks = chunk_text(
            _metin(20), max_tokens=30, overlap_sentences=1, token_counter=word_count
        )
        for prev, nxt in pairwise(chunks):
            assert nxt.first_sentence == prev.last_sentence

    def test_ortusme_kapatilabilir(self):
        chunks = chunk_text(
            _metin(20), max_tokens=30, overlap_sentences=0, token_counter=word_count
        )
        for prev, nxt in pairwise(chunks):
            assert nxt.first_sentence == prev.last_sentence + 1

    def test_ortusme_butceyi_delemez(self):
        """Regresyon: devralınan cümle sonrası bütçe kontrol edilmiyordu.

        Parça dolunca son cümle bir sonrakine devrediliyordu ama devirden
        sonra bütçe kontrolü yoktu; yeni parça daha doğarken sınırı aşıyordu.
        """
        metin = "A b c d e f g h. I j k l m n o p q r s t u v w x."  # 8 + 16 kelime
        chunks = chunk_text(metin, max_tokens=20, overlap_sentences=1, token_counter=word_count)
        assert [word_count(c.text) for c in chunks] == [8, 16]
        assert chunks[1].first_sentence == 1  # örtüşmeden vazgeçildi

    def test_ortusme_sigiyorsa_korunur(self):
        metin = "A b c. D e f g h i j k l m n o p q. R s t."  # 3 + 14 + 3 kelime
        chunks = chunk_text(metin, max_tokens=18, overlap_sentences=1, token_counter=word_count)
        assert [word_count(c.text) for c in chunks] == [17, 17]
        assert chunks[1].text.startswith("D e f")  # önceki parçanın son cümlesi

    def test_son_parca_butceyi_asmaz(self):
        """Regresyon: küçük son parça %15 toleransla öncekine katılıyordu.

        18 + 4 = 22 kelime, bütçe 20. Toleranslı birleştirme 22'lik bir parça
        üretirdi — tam da önlemeye çalıştığımız taşma.
        """
        metin = "A b c d e f g h i j k l m n o p q r. S t u v."
        chunks = chunk_text(metin, max_tokens=20, overlap_sentences=0, token_counter=word_count)
        assert [word_count(c.text) for c in chunks] == [18, 4]

    def test_butceyi_asan_tek_cumle_kelimeden_bolunur(self):
        uzun = " ".join(f"k{i}" for i in range(50)) + "."
        chunks = chunk_text(uzun, max_tokens=20, overlap_sentences=0, token_counter=word_count)
        assert all(c.token_count <= 20 for c in chunks)
        assert " ".join(c.text for c in chunks) == uzun

    def test_varsayilan_sayac_sezgisel(self):
        chunks = chunk_text(_metin(100), max_tokens=100)
        assert all(heuristic_token_count(c.text) <= 100 for c in chunks)

    def test_gercek_tokenizer_kullanilir(self):
        cagrilar: list[str] = []

        def sayac(metin: str) -> int:
            cagrilar.append(metin)
            return len(metin)

        chunks = chunk_text("Bir iki üç. Dört beş altı.", max_tokens=15, token_counter=sayac)
        assert cagrilar
        assert all(c.token_count == len(c.text) <= 15 for c in chunks)

    @pytest.mark.parametrize(("butce", "ortusme"), [(0, 1), (-5, 1), (10, -1)])
    def test_gecersiz_parametre(self, butce, ortusme):
        with pytest.raises(ValueError):
            chunk_text("Metin.", max_tokens=butce, overlap_sentences=ortusme)

    @pytest.mark.parametrize("tohum", range(25))
    def test_rastgele_metinlerde_garantiler(self, tohum):
        """Elle yazılmış örneklerin kaçırdığı sınır durumları için."""
        rng = random.Random(tohum)
        cumleler = [
            " ".join(f"w{i}x{j}" for j in range(rng.randint(1, 30))) + "."
            for i in range(rng.randint(1, 80))
        ]
        metin = " ".join(f"Ç{c}" for c in cumleler)
        butce = rng.randint(8, 60)
        ortusme = rng.randint(0, 3)

        chunks = chunk_text(
            metin, max_tokens=butce, overlap_sentences=ortusme, token_counter=word_count
        )

        assert all(word_count(c.text) <= butce for c in chunks)
        birlesik = " ".join(c.text for c in chunks)
        for kelime in metin.split():
            assert kelime in birlesik
        starts = [c.first_sentence for c in chunks]
        assert starts == sorted(starts)
