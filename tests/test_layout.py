"""Sayfa süslemesi ve kaynakça kesimi testleri.

Yanlış silme geri getirilemez; bu yüzden testlerin çoğu güvenlik
kurallarını (uzun satır, gövde içi tekrar, kısa sayfa, kesim freni)
doğrular.
"""

from __future__ import annotations

import pytest

from ozetle.pdf.layout import (
    cut_references,
    find_repeating_lines,
    is_page_number,
    strip_page_furniture,
)

GOVDE = (
    "Bu sayfanın benzersiz içeriği {i} burada yer almaktadır ve gövde metni normal bir "
    "paragraf uzunluğunda olduğu için süsleme bandının dışında kalmalıdır."
)


def _sayfalar(n: int = 6) -> list[str]:
    """Gerçek bir makale sayfası: kısa header, uzun gövde, kısa footer."""
    return [
        f"Fizyoterapi Dergisi 2019; 30(1)\n{GOVDE.format(i=i)}\nSayfa {i}\n{i}"
        for i in range(1, n + 1)
    ]


class TestIsPageNumber:
    @pytest.mark.parametrize(
        "satir",
        [
            "12",
            "- 12 -",
            "\u2013 7 \u2013",
            "12 / 30",
            "Sayfa 3",
            "SAYFA 3",
            "Page 4",
            "Page 4 of 10",
            "s. 45",
        ],
    )
    def test_sayfa_numarasi(self, satir):
        assert is_page_number(satir)

    @pytest.mark.parametrize(
        "satir", ["2019 yılında", "12 hasta katıldı", "Tablo 3", "45 mg", "", "2024", "1998"]
    )
    def test_sayfa_numarasi_degil(self, satir):
        assert not is_page_number(satir)


class TestStripPageFurniture:
    def test_header_silinir(self):
        birlesik = "\n".join(strip_page_furniture(_sayfalar()))
        assert "Fizyoterapi Dergisi" not in birlesik

    def test_sayfa_numarasi_silinir(self):
        birlesik = "\n".join(strip_page_furniture(_sayfalar()))
        assert "Sayfa 3" not in birlesik
        assert "\n3\n" not in f"\n{birlesik}\n"

    def test_govde_korunur(self):
        temiz = strip_page_furniture(_sayfalar())
        for i, sayfa in enumerate(temiz, start=1):
            assert sayfa == GOVDE.format(i=i)

    def test_degisen_sayilar_ayni_suslemedir(self):
        """'Sayfa 3' ve 'Sayfa 4' aynı satırdır; sayılar karşılaştırmada eşitlenir."""
        anahtarlar = find_repeating_lines(_sayfalar())
        assert "sayfa #" in anahtarlar

    def test_uzun_satir_kenarda_bile_silinmez(self):
        """Uzunluk koruması: gövde metni asla süsleme sayılmaz."""
        uzun = "A" * 150
        sayfalar = [f"{uzun}\nOrta içerik {i}\n{uzun}" for i in range(1, 6)]
        birlesik = "\n".join(strip_page_furniture(sayfalar))
        assert birlesik.count(uzun) == 10

    def test_govde_icindeki_tekrar_silinmez(self):
        """Dergi adı gövdede geçiyorsa (atıf gibi) kenarda olmadığı için kalır."""
        sayfalar = [
            "Fizyoterapi Dergisi 2019; 30(1)\n"
            f"{GOVDE.format(i=i)}\n"
            "Fizyoterapi Dergisi 2019; 30(1)\n"
            f"{GOVDE.format(i=i + 100)}\n{i}"
            for i in range(1, 6)
        ]
        temiz = strip_page_furniture(sayfalar)
        assert all(s.count("Fizyoterapi Dergisi") == 1 for s in temiz)

    def test_kisa_sayfa_tamamen_silinmez(self):
        """Regresyon: sabit kenar penceresi kısa sayfanın tamamını siliyordu.

        "2024" ayrıca yıl korumasını doğrular: diğer sayfalardaki çıplak
        sayfa numaralarıyla aynı anahtara düşüp silinmemeli.
        """
        sayfalar = _sayfalar(5)
        sayfalar.insert(0, "Başlık Sayfası\nYazar Adı\nKurum\n2024")
        temiz = strip_page_furniture(sayfalar)
        assert temiz[0] == "Başlık Sayfası\nYazar Adı\nKurum\n2024"

    def test_az_sayfada_tekrar_tespiti_yapilmaz(self):
        """İki sayfada tekrar eden satır tesadüf olabilir."""
        sayfalar = ["Ortak satır\nİçerik bir\n1", "Ortak satır\nİçerik iki\n2"]
        temiz = strip_page_furniture(sayfalar)
        assert all(s.startswith("Ortak satır") for s in temiz)
        assert all(not s.endswith(("1", "2")) for s in temiz)  # numara yine de gider

    def test_bos_satirlar_kenar_bandini_bozmaz(self):
        sayfalar = [f"\n\nHeader Satırı\n\n{GOVDE.format(i=i)}\n\n{i}\n" for i in range(1, 6)]
        birlesik = "\n".join(strip_page_furniture(sayfalar))
        assert "Header Satırı" not in birlesik
        assert birlesik.count("benzersiz içeriği") == 5

    def test_bos_girdi(self):
        assert strip_page_furniture([]) == []


class TestCutReferences:
    @staticmethod
    def _belge(baslik: str) -> str:
        govde = "\n".join(f"Bu {i}. paragraftır ve anlamlı içerik taşır." for i in range(30))
        return (
            f"{govde}\n{baslik}\n"
            "[1] Yılmaz A. Bir makale. Dergi. 2019.\n"
            "[2] Demir B. Başka makale. Dergi. 2020.\n"
            "[3] Kaya C. Üçüncü makale. Dergi. 2021."
        )

    @pytest.mark.parametrize(
        "baslik",
        [
            "KAYNAKLAR",
            "Kaynakça",
            "KAYNAKÇA",
            "Kaynaklar:",
            "REFERENCES",
            "Referanslar",
            "7. KAYNAKLAR",
            "Bibliography",
        ],
    )
    def test_kaynakca_kesilir(self, baslik):
        out = cut_references(self._belge(baslik))
        assert "Yılmaz A." not in out
        assert baslik not in out
        assert "29. paragraftır" in out

    def test_govde_icindeki_kaynaklar_kelimesi_kesmez(self):
        text = "Kaynaklar gösteriyor ki bu doğrudur. " * 20
        assert cut_references(text) == text

    def test_baslik_yoksa_degismez(self):
        text = "Kaynakça içermeyen normal bir metin."
        assert cut_references(text) == text

    def test_guvenlik_freni(self):
        """Kesim metnin çoğunu uçuruyorsa iptal edilmeli."""
        text = "Kısa giriş.\nKAYNAKLAR\n" + "\n".join(
            f"[{i}] Yazar {i}. Makale. 2019." for i in range(1, 40)
        )
        assert cut_references(text) == text

    def test_son_baslik_kullanilir(self):
        """İçindekiler tablosundaki 'Kaynakça' satırı belgenin başını kesmemeli."""
        belge = "İÇİNDEKİLER\nGiriş\nKaynakça\n" + self._belge("KAYNAKÇA")
        out = cut_references(belge)
        assert out.startswith("İÇİNDEKİLER")
        assert "29. paragraftır" in out
        assert "Yılmaz A." not in out
