"""PDF çıkarma testleri — gerçek PDF dosyaları üzerinde.

Fixture'lar :mod:`pdf_fixtures` ile, Türkçe karakter taşıyabilen gömülü
DejaVu fontuyla üretilir.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ozetle.pdf import extract as extract_module
from ozetle.pdf.extract import (
    EmptyPdfError,
    EncryptedPdfError,
    InvalidPdfError,
    PdfDocument,
    PdfPage,
    PdfTooLargeError,
    extract,
    find_gutter,
    measure_garbling,
    strip_cid_artifacts,
    words_to_text,
)
from pdf_fixtures import (
    make_article_pdf,
    make_blank_pdf,
    make_pdf,
    make_two_column_pdf,
    wrap,
)

TURKCE = "Çalışma İstanbul'da yapılmıştır; ölçümler ğ, ş, ı ve ü harflerini içerir."


@pytest.fixture
def iki_sutun(tmp_path: Path) -> Path:
    left = wrap("Sol sütundaki birinci paragraf yapılmıştır ve devam etmektedir. " * 6, 42)
    right = wrap("Sağ sütundaki ikinci paragraf ölçülmüştür ve sonuçlanmıştır. " * 6, 42)
    return make_two_column_pdf(
        tmp_path / "iki.pdf",
        title="İki Sütunlu Bir Başlık Örneği Çalışması",
        left=left,
        right=right,
    )


class TestTurkceKarakterler:
    def test_turkce_karakterler_bozulmadan_cikar(self, tmp_path):
        doc = extract(make_pdf(tmp_path / "tr.pdf", [[TURKCE]]))
        assert doc.pages[0].text == TURKCE

    def test_helvetica_turkce_tasiyamaz(self, tmp_path):
        """Fixture tuzağının belgesi: Helvetica ile yazılan Türkçe bozulur.

        İlk fixture'lar böyle yazılmıştı ve testler geçiyordu — Türkçe test
        ettiğimizi sanıyorduk. Bu test, fixture'ların neden DejaVu kullandığını
        kalıcı olarak kayıt altına alır.
        """
        doc = extract(make_pdf(tmp_path / "hv.pdf", [[TURKCE]], font="Helvetica"))
        assert doc.pages[0].text != TURKCE
        assert "yapılmıştır" not in doc.pages[0].text


class TestBelgeYapisi:
    def test_sayfalar_ve_metadata(self, tmp_path):
        doc = extract(make_article_pdf(tmp_path / "m.pdf", n_body_pages=3))
        assert doc.total_pages == doc.pages_read == 4
        assert [p.number for p in doc.pages] == [1, 2, 3, 4]
        assert doc.metadata["Title"] == "Test Makalesi"
        assert not doc.truncated

    def test_sayfa_siniri(self, tmp_path):
        doc = extract(make_article_pdf(tmp_path / "m.pdf", n_body_pages=5), max_pages=2)
        assert doc.pages_read == 2
        assert doc.total_pages == 6
        assert doc.truncated

    def test_bayt_girdisi(self, tmp_path):
        data = make_pdf(tmp_path / "b.pdf", [[TURKCE]]).read_bytes()
        assert extract(data).pages[0].text == TURKCE

    def test_char_count_birlestirilmis_metindir(self, tmp_path):
        doc = extract(make_pdf(tmp_path / "c.pdf", [["abc"], ["defg"]]))
        assert doc.char_count == len("abc\ndefg")


class TestSutunTespiti:
    def test_iki_sutun_tespit_edilir(self, iki_sutun):
        assert extract(iki_sutun).pages[0].columns == 2

    def test_okuma_sirasi_sol_sonra_sag(self, iki_sutun):
        text = extract(iki_sutun).pages[0].text
        lines = text.splitlines()
        sol = [i for i, ln in enumerate(lines) if "Sol" in ln or "birinci" in ln]
        sag = [i for i, ln in enumerate(lines) if "Sağ" in ln or "ikinci" in ln]
        assert max(sol) < min(sag)

    def test_baslik_en_ustte_ve_butun(self, iki_sutun):
        """Regresyon: başlığın son kelimesi sağ sütunun başına kayıyordu."""
        text = extract(iki_sutun).pages[0].text
        assert text.splitlines()[0] == "İki Sütunlu Bir Başlık Örneği Çalışması"

    def test_tespit_kapaliyken_satirlar_ic_ice_gecer(self, iki_sutun):
        """Tespitin neden gerektiğinin kanıtı: naif okuma satırları karıştırır."""
        text = extract(iki_sutun, detect_columns=False).pages[0].text
        karisik = [ln for ln in text.splitlines() if "Sol" in ln and "Sağ" in ln]
        assert karisik

    def test_tek_sutunda_yanlis_pozitif_yok(self, tmp_path):
        """Regresyon: benzer satırlarda kelime araları alt alta hizalanıp
        sahte bir sütun boşluğu oluşturuyordu ve her satır ortadan bölünüyordu."""
        pdf = make_article_pdf(tmp_path / "m.pdf", n_body_pages=3)
        acik, kapali = extract(pdf), extract(pdf, detect_columns=False)
        assert all(p.columns == 1 for p in acik.pages)
        assert acik.texts() == kapali.texts()

    def test_az_kelimede_tespit_yapilmaz(self):
        words = [{"text": "a", "x0": 10.0, "x1": 20.0, "top": 10.0, "bottom": 20.0}] * 5
        assert find_gutter(words, 600.0) is None

    def test_alt_alta_bloklar_sutun_sayilmaz(self):
        """Solda üst blok, sağda alt blok: yan yana değiller, sütun değil."""
        words = [
            {
                "text": f"s{i}",
                "x0": 50.0,
                "x1": 200.0,
                "top": 100.0 + i * 12,
                "bottom": 110.0 + i * 12,
            }
            for i in range(15)
        ] + [
            {
                "text": f"r{i}",
                "x0": 350.0,
                "x1": 500.0,
                "top": 400.0 + i * 12,
                "bottom": 410.0 + i * 12,
            }
            for i in range(15)
        ]
        assert find_gutter(words, 600.0) is None
        text, columns = words_to_text(words, 600.0, detect_columns=True)
        assert columns == 1
        assert text.splitlines()[0] == "s0"


class TestHatalar:
    def test_dosya_yok(self, tmp_path):
        with pytest.raises(InvalidPdfError):
            extract(tmp_path / "yok.pdf")

    def test_pdf_degil(self, tmp_path):
        sahte = tmp_path / "sahte.pdf"
        sahte.write_text("bu bir metin dosyası")
        with pytest.raises(InvalidPdfError):
            extract(sahte)

    def test_bozuk_pdf(self, tmp_path):
        bozuk = tmp_path / "bozuk.pdf"
        bozuk.write_bytes(b"%PDF-1.4\n" + b"\x00garbage" * 50)
        with pytest.raises(InvalidPdfError):
            extract(bozuk)

    def test_boyut_siniri_dosya(self, tmp_path):
        pdf = make_pdf(tmp_path / "b.pdf", [[TURKCE]])
        with pytest.raises(PdfTooLargeError):
            extract(pdf, max_bytes=100)

    def test_boyut_siniri_bayt(self, tmp_path):
        data = make_pdf(tmp_path / "b.pdf", [[TURKCE]]).read_bytes()
        with pytest.raises(PdfTooLargeError):
            extract(data, max_bytes=100)

    def test_parolali_pdf(self, tmp_path):
        pdf = make_pdf(tmp_path / "p.pdf", [[TURKCE]], encrypt="gizli")
        with pytest.raises(EncryptedPdfError):
            extract(pdf)

    def test_taranmis_pdf(self, tmp_path):
        with pytest.raises(EmptyPdfError, match="OCR"):
            extract(make_blank_pdf(tmp_path / "bos.pdf"))

    def test_gecersiz_sayfa_siniri(self, tmp_path):
        with pytest.raises(ValueError):
            extract(make_pdf(tmp_path / "x.pdf", [["a"]]), max_pages=0)


class TestKodlamaKalitesi:
    def test_temiz_metin(self):
        assert measure_garbling(TURKCE) == 0.0
        assert measure_garbling("") == 0.0

    def test_cid_artiklari_olculur(self):
        bozuk = "Bu (cid:12)(cid:45)al(cid:33)(cid:91)ma yap(cid:12)lm(cid:12)(cid:15)t(cid:12)r."
        assert measure_garbling(bozuk) > 0.5

    def test_degistirme_karakteri_olculur(self):
        assert measure_garbling("ab�") == pytest.approx(1 / 3)

    def test_oran_bir_ile_sinirli(self):
        assert measure_garbling("(cid:1)") == 1.0

    def test_artiklar_temizlenir(self):
        assert strip_cid_artifacts("a(cid:12)b�c") == "abc"

    def test_esik(self):
        assert not PdfDocument(pages=[], total_pages=0, garbled_ratio=0.01).has_encoding_issues
        assert PdfDocument(pages=[], total_pages=0, garbled_ratio=0.05).has_encoding_issues

    def test_bozuk_kodlama_uyarilir_ama_engellenmez(self, tmp_path, monkeypatch, caplog):
        """Gömülü olmayan font: hata değil uyarı. İş devam eder, kullanıcı bilgilenir."""
        pdf = make_pdf(tmp_path / "x.pdf", [["a"], ["b"]])
        monkeypatch.setattr(
            extract_module,
            "_page_text",
            lambda page, detect_columns: ("Bu (cid:12)al(cid:33)(cid:91)ma t(cid:12)r.", 1),
        )
        with caplog.at_level(logging.WARNING, logger="ozetle.pdf.extract"):
            doc = extract(pdf)
        assert doc.has_encoding_issues
        assert "(cid:" not in doc.pages[0].text
        assert "kodlaması bozuk" in caplog.text

    def test_sayfa_nesnesi_degismez(self):
        page = PdfPage(number=1, text="x")
        with pytest.raises(AttributeError):
            page.text = "y"  # type: ignore[misc]
