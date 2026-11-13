"""Uçtan uca hat testleri: PDF girer, modele hazır parçalar çıkar."""

from __future__ import annotations

import json

import pytest

from ozetle.__main__ import main
from ozetle.pdf.extract import PdfDocument, PdfPage
from ozetle.pipeline import prepare, prepare_document
from pdf_fixtures import JOURNAL_HEADER, make_article_pdf, make_pdf, make_two_column_pdf, wrap


@pytest.fixture
def makale(tmp_path):
    return make_article_pdf(tmp_path / "makale.pdf", n_body_pages=6)


def kelime(metin: str) -> int:
    return len(metin.split())


class TestPrepare:
    def test_susleme_ve_kaynakca_gider(self, makale):
        doc = prepare(makale)
        assert JOURNAL_HEADER not in doc.text
        assert "Sayfa 3" not in doc.text
        assert "KAYNAKLAR" not in doc.text
        assert "Yazar1 A." not in doc.text

    def test_bulgular_modele_ulasir(self, makale):
        """Eski hat 512 token'dan sonrasını atıyor, parantezli sayıları siliyordu."""
        doc = prepare(makale)
        birlesik = " ".join(c.text for c in doc.chunks)
        for i in range(1, 19):
            assert f"Bulgu {i}:" in birlesik
            assert f"Tablo {i} " in birlesik
        assert "(p < 0.05)" in birlesik
        assert "(n = 45)" in birlesik
        assert "(Demir ve ark., 2019)" in birlesik

    def test_parcalar_butceye_sigar(self, makale):
        doc = prepare(makale, max_tokens=120, token_counter=kelime)
        assert len(doc.chunks) > 1
        assert all(kelime(c.text) <= 120 for c in doc.chunks)

    def test_metrikler_monoton_azalir(self, makale):
        """Regresyon: aşamalar farklı biçimde ölçülüyordu.

        Süsleme sonrası birleştirme öncesi, kaynakça sonrası birleştirme
        sonrası sayılınca kesme adımı karakter eklemiş görünüyordu
        (süsleme 3191 → kaynakça 3194).
        """
        m = prepare(makale).metrics
        assert m.raw >= m.after_furniture >= m.after_references
        assert m.furniture_removed > 0
        assert m.references_removed > 0
        assert set(m.as_dict()) == {"raw", "after_furniture", "after_references", "after_clean"}

    def test_sezgisel_sayac_uyarisi(self, makale):
        assert any("tokenizer" in w for w in prepare(makale).warnings)
        assert not any("tokenizer" in w for w in prepare(makale, token_counter=kelime).warnings)

    def test_kesik_belge_uyarisi(self, makale):
        doc = prepare(makale, max_pages=2)
        assert doc.pages_read == 2
        assert any("ilk 2 sayfa" in w for w in doc.warnings)

    def test_bayt_girdisi(self, makale):
        assert prepare(makale.read_bytes()).text == prepare(makale).text

    def test_iki_sutunlu_belgede_cumleler_butun(self, tmp_path):
        cumle = "Sol sütundaki birinci paragraf yapılmıştır ve devam etmektedir."
        pdf = make_two_column_pdf(
            tmp_path / "iki.pdf",
            title="Başlık",
            left=wrap(f"{cumle} " * 6, 42),
            right=wrap("Sağ sütundaki ikinci paragraf ölçülmüştür ve sonuçlanmıştır. " * 6, 42),
        )
        assert prepare(pdf).text.count(cumle) == 6
        assert prepare(pdf, detect_columns=False).text.count(cumle) < 6


class TestPrepareDocument:
    def test_bozuk_kodlama_uyarisi(self):
        document = PdfDocument(
            pages=[PdfPage(number=1, text="Bir cümle.")], total_pages=1, garbled_ratio=0.3
        )
        doc = prepare_document(document)
        assert any("font kodlaması" in w for w in doc.warnings)
        assert doc.garbled_ratio == 0.3

    def test_her_sey_susleme_ise_bos_kalir(self):
        document = PdfDocument(
            pages=[PdfPage(number=i, text=str(i)) for i in range(1, 5)], total_pages=4
        )
        doc = prepare_document(document)
        assert doc.is_empty
        assert any("kalmadı" in w for w in doc.warnings)


class TestCli:
    def test_ozet_cikti(self, makale, capsys):
        assert main([str(makale)]) == 0
        out = capsys.readouterr().out
        assert "Sayfa          : 7/7" in out
        assert "Parça" in out

    def test_json_cikti(self, makale, capsys):
        assert main([str(makale), "--json", "--max-tokens", "200"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["pages"] == {"total": 7, "read": 7}
        assert payload["chunks"]
        assert all(c["tokens"] <= 200 for c in payload["chunks"])

    def test_parcalari_gosterir(self, makale, capsys):
        assert main([str(makale), "--show-chunks", "--no-columns"]) == 0
        assert "--- parça 0" in capsys.readouterr().out

    def test_hata_kodu(self, tmp_path, capsys):
        sahte = make_pdf(tmp_path / "x.pdf", [["a"]])
        sahte.write_text("PDF değil")
        assert main([str(sahte)]) == 1
        assert "hata:" in capsys.readouterr().err
