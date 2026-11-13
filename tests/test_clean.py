"""Metin temizleme testleri.

Ana fikir: temizlik muhafazakâr olmalı. Buradaki testlerin yarısı bir şeyin
silindiğini değil, **silinmediğini** doğrular \u2014 eski kodun en büyük hatası
bulguları gürültü sanıp silmesiydi.
"""

from __future__ import annotations

import unicodedata

import pytest

from ozetle.text.clean import (
    clean,
    dehyphenate,
    normalize_unicode,
    normalize_whitespace,
    remove_citation_markers,
    unwrap_lines,
)


class TestNormalizeUnicode:
    def test_ligaturler_acilir(self):
        assert normalize_unicode("efﬁcient ﬂow") == "efficient flow"
        assert normalize_unicode("eﬃcient") == "efficient"

    def test_gorunmez_karakterler_silinir(self):
        assert normalize_unicode("de­ğer​len﻿dirme") == "değerlendirme"

    def test_kirilmaz_bosluk_duz_bosluk_olur(self):
        assert normalize_unicode("500\u00a0mg") == "500 mg"

    def test_ayrisik_turkce_karakter_birlesir(self):
        ayrisik = unicodedata.normalize("NFD", "çalışması")
        assert ayrisik != "çalışması"
        assert normalize_unicode(ayrisik) == "çalışması"

    def test_noktali_i_birlesimi_duz_i_olur(self):
        """'İ'.lower() → 'i̇' (i + U+0307); kelime karşılaştırmasını bozar."""
        assert "İstanbul".lower() != "istanbul"
        assert normalize_unicode("İstanbul".lower()) == "istanbul"

    def test_turkce_harfler_korunur(self):
        metin = "ÇĞİÖŞÜ çğıöşü"
        assert normalize_unicode(metin) == metin


class TestDehyphenate:
    def test_kucuk_harfle_devam_eden_satir_birlesir(self):
        assert dehyphenate("değerlendir-\nmesi yapıldı") == "değerlendirmesi yapıldı"

    def test_turkce_kucuk_harfle_devam(self):
        assert dehyphenate("yapıl-\nmıştır") == "yapılmıştır"
        assert dehyphenate("çalış-\nması") == "çalışması"

    def test_buyuk_harf_veya_rakamla_devam_eden_tire_korunur(self):
        assert dehyphenate("COVID-\n19 salgını") == "COVID-\n19 salgını"
        assert dehyphenate("Ankara-\nİstanbul hattı") == "Ankara-\nİstanbul hattı"

    def test_satir_ici_tire_dokunulmaz(self):
        assert dehyphenate("sosyo-ekonomik durum") == "sosyo-ekonomik durum"


class TestRemoveCitationMarkers:
    @pytest.mark.parametrize(
        ("girdi", "beklenen"),
        [
            ("Etkili bulundu [3].", "Etkili bulundu."),
            ("Etkili bulundu [3, 4].", "Etkili bulundu."),
            ("Etkili bulundu [4-7].", "Etkili bulundu."),
            ("Etkili bulundu [4\u20137].", "Etkili bulundu."),
            ("Etkili bulundu [1,3-5].", "Etkili bulundu."),
        ],
    )
    def test_numarali_atif_silinir(self, girdi, beklenen):
        assert remove_citation_markers(girdi) == beklenen

    @pytest.mark.parametrize(
        "korunacak",
        [
            "(p < 0.05)",
            "(n = 45)",
            "(Demir ve ark., 2019)",
            "500 mg (2 doz)",
            "[2024]",  # dört basamaklı yıl atıf değildir
            "Tablo 3",
            "Şekil 2",
        ],
    )
    def test_bulgular_silinmez(self, korunacak):
        """Eski kod sayı içeren her parantezi siliyordu \u2014 p-değerleri dahil."""
        assert korunacak in remove_citation_markers(f"Sonuç {korunacak} görüldü.")


class TestUnwrapAndWhitespace:
    def test_tek_satir_sonu_bosluk_olur(self):
        assert unwrap_lines("bir cümle\nikinci satırda devam") == "bir cümle ikinci satırda devam"

    def test_paragraf_siniri_korunur(self):
        assert unwrap_lines("birinci\nparagraf\n\nikinci\nparagraf") == (
            "birinci paragraf\n\nikinci paragraf"
        )

    def test_bos_paragraflar_atilir(self):
        assert unwrap_lines("a\n\n\n\n   \n\nb") == "a\n\nb"

    def test_noktalama_oncesi_bosluk_toplanir(self):
        assert normalize_whitespace("sonuç  bulundu .") == "sonuç bulundu."


class TestClean:
    def test_bos_girdi(self):
        assert clean("") == ""

    def test_uctan_uca(self):
        ham = (
            "Hastaların fonksiyonel skorlarında an-\nlamlı iyileşme gözlendi [3].\n"
            "Tablo 2 sonuçları özetler (p < 0.05).\n\n"
            "İkinci paraﬁ  graf."
        )
        assert clean(ham) == (
            "Hastaların fonksiyonel skorlarında anlamlı iyileşme gözlendi. "
            "Tablo 2 sonuçları özetler (p < 0.05).\n\n"
            "İkinci parafi graf."
        )

    def test_atif_silme_kapatilabilir(self):
        assert "[3]" in clean("Bulgu [3].", remove_citations=False)

    def test_windows_satir_sonlari(self):
        assert clean("bir\r\niki\rüç") == "bir iki üç"

    def test_kelime_kara_listesi_yok(self):
        """'Tablo', 'Şekil' gibi kelimeler kör silinmez."""
        metin = "Tablo 1 ve Şekil 2 karşılaştırıldı."
        assert clean(metin) == metin


class TestHeadings:
    GOVDE = "Bu çalışmada diz osteoartriti tanısı almış yetişkinlerde on iki haftalık"

    def test_noktalamasiz_baslik_ayri_paragraf(self) -> None:
        metin = f"Özet\n{self.GOVDE}\nprogram incelenmiştir.\nYöntem\nÇalışmaya katılmıştır."
        assert unwrap_lines(metin).split("\n\n") == [
            "Özet",
            f"{self.GOVDE} program incelenmiştir.",
            "Yöntem",
            "Çalışmaya katılmıştır.",
        ]

    def test_uzun_govde_satiri_ozel_isimle_devam_etse_bolunmez(self) -> None:
        metin = f"{self.GOVDE} programı Ankara'da\nAnkara Üniversitesi hastanesinde yürütüldü."
        assert "\n\n" not in unwrap_lines(metin)

    def test_noktayla_biten_kisa_satir_baslik_degildir(self) -> None:
        metin = f"{self.GOVDE} çalışma.\nKısa son satır.\nYeni paragraf başlıyor burada."
        assert "\n\n" not in unwrap_lines(metin)

    def test_kucuk_harfle_baslayan_kisa_satir_baslik_degildir(self) -> None:
        metin = f"{self.GOVDE} programın ağrı ve işlev\nüzerine etkisi\nÖzet\nBu çalışma."
        assert unwrap_lines(metin).split("\n\n")[0].endswith("üzerine etkisi")

    def test_cumle_bolme_baslikta_ayrilir(self) -> None:
        from ozetle.text.chunk import split_sentences

        metin = clean(f"Özet\n{self.GOVDE}\nprogram incelenmiştir.")
        assert split_sentences(metin)[0] == "Özet"
