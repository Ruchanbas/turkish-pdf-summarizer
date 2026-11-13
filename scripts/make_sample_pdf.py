"""Deneme için örnek Türkçe makale PDF'i üretir: docs/ornek-makale.pdf

Metin bu proje için yazılmış **kurgusal** bir çalışmadır; kişiler, kurumlar
ve sayılar gerçek değildir. Hattın göstermesi gereken her şeyi içerir:
her sayfada tekrar eden dergi başlığı ve sayfa numarası (silinmeli),
bulgulardaki parantezli sayılar (korunmalı) ve sonda kaynakça (kesilmeli).

    python scripts/make_sample_pdf.py docs/ornek-makale.pdf
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "tests" / "fixtures" / "fonts" / "DejaVuSans.ttf"

HEADER = "Kurgusal Rehabilitasyon Bilimleri Dergisi 2026; 4(2)"

TITLE = "Diz osteoartritinde denetimli egzersiz programının ağrı ve işlev üzerine etkisi"

SECTIONS: list[tuple[str, str]] = [
    (
        "Özet",
        "Bu çalışmada diz osteoartriti tanısı almış yetişkinlerde on iki haftalık denetimli "
        "egzersiz programının ağrı, fonksiyonel kapasite ve yaşam kalitesi üzerindeki etkisi "
        "incelenmiştir. Program, kuvvetlendirme, denge ve aerobik egzersizleri bir araya "
        "getirmiştir. Katılımcılar egzersiz grubu ve yalnızca eğitim alan kontrol grubu olarak "
        "rastgele iki gruba ayrılmıştır.",
    ),
    (
        "Giriş",
        "Diz osteoartriti, orta ve ileri yaştaki yetişkinlerde ağrının ve hareket kısıtlılığının "
        "en sık nedenlerinden biridir. Hastalık ilerledikçe kişiler merdiven çıkmak, uzun süre "
        "yürümek ve yerden kalkmak gibi günlük işlerde zorlanır. Klinik kılavuzlar egzersizi "
        "tedavinin temel taşı olarak önerse de hangi egzersiz bileşeninin ne kadar katkı "
        "sağladığı tartışmalıdır. Özellikle kuadriseps kas kuvveti ile ağrı arasındaki ilişki, "
        "programların tasarımında belirleyici bir rol oynar. Bu çalışmanın amacı, kuvvetlendirme "
        "ağırlıklı denetimli bir programın ağrı ve işlev üzerindeki etkisini ölçmek ve bu etkinin "
        "kas kuvvetindeki değişimle ne ölçüde açıklanabildiğini incelemektir.",
    ),
    (
        "Yöntem",
        "Çalışmaya 50 ile 75 yaş arasında, en az altı aydır diz ağrısı yaşayan 64 gönüllü "
        "katılmıştır. Egzersiz grubu (n = 32) haftada üç gün, fizyoterapist gözetiminde 45 "
        "dakikalık seanslara katılmıştır. Seanslar ısınma, kuadriseps ve kalça abdüktör "
        "kuvvetlendirmesi, tek ayak üzerinde denge çalışması ve düşük yoğunluklu bisiklet "
        "egzersizinden oluşmuştur. Kontrol grubu (n = 32) hastalık hakkında iki eğitim oturumuna "
        "katılmış ve günlük yaşamına devam etmiştir. Ağrı görsel analog skala ile, fonksiyonel "
        "kapasite otur-kalk testi ve altı dakika yürüme testi ile, kas kuvveti ise el "
        "dinamometresi ile ölçülmüştür. Değerlendirmeler başlangıçta ve on ikinci haftanın "
        "sonunda, grup ataması bilinmeyen bir değerlendirici tarafından yapılmıştır.",
    ),
    (
        "Bulgular",
        "On iki haftanın sonunda egzersiz grubunda ağrı skoru ortalama 6,8'den 3,9'a düşmüş, "
        "kontrol grubunda ise 6,6'dan 6,1'e gerilemiştir (p < 0.001). Altı dakika yürüme "
        "mesafesi egzersiz grubunda ortalama 58 metre artarken kontrol grubunda anlamlı bir "
        "değişim görülmemiştir. Kuadriseps kas kuvvetindeki artış egzersiz grubunda yüzde 21 "
        "olarak ölçülmüş ve ağrıdaki azalmanın yaklaşık üçte birini açıklamıştır (r = 0.46). "
        "Tablo 2'de görüldüğü gibi otur-kalk testi süresi de egzersiz grubunda belirgin biçimde "
        "kısalmıştır. Programa devam oranı yüzde 88 olmuş, ciddi bir yan etki bildirilmemiştir.",
    ),
    (
        "Tartışma",
        "Bulgular, denetimli ve kuvvetlendirme ağırlıklı bir egzersiz programının diz "
        "osteoartritinde ağrıyı azaltıp işlevi artırdığını göstermektedir. Kas kuvvetindeki "
        "artışın etkinin yalnızca bir bölümünü açıklaması, denge çalışmalarının ve düzenli "
        "fiziksel aktivitenin de katkı sağladığını düşündürmektedir. Yüksek devam oranı, "
        "fizyoterapist gözetiminin programa bağlılığı desteklediğine işaret eder. Çalışmanın "
        "sınırlılıkları arasında izlem süresinin kısa olması ve katılımcıların tek bir "
        "merkezden seçilmesi sayılabilir.",
    ),
    (
        "Sonuç",
        "On iki haftalık denetimli egzersiz programı, diz osteoartriti olan yetişkinlerde ağrıyı "
        "anlamlı ölçüde azaltmış ve yürüme kapasitesini artırmıştır. Kuvvetlendirme, denge ve "
        "aerobik bileşenlerin birlikte uygulanması önerilir; uzun dönem etkilerin daha geniş "
        "örneklemlerle araştırılması gereklidir.",
    ),
]

REFERENCES = [
    "Kaya A, Demir B. Diz osteoartritinde egzersiz yaklaşımları. Kurgusal Derg. 2021;2:10-18.",
    "Yıldız C. Kas kuvveti ve ağrı ilişkisi. Kurgusal Fizyoter Arş. 2022;5:44-52.",
    "Aydın D, Şahin E. Denetimli programlarda devamlılık. Kurgusal Rehabil. 2023;7:3-9.",
    "Öztürk F. Fonksiyonel testlerin güvenirliği. Kurgusal Ölçüm Derg. 2020;1:21-30.",
]


def build(out: Path) -> Path:
    pdfmetrics.registerFont(TTFont("DejaVu", str(FONT)))
    _, height = A4
    margin, line_h, size = 72, 22, 12
    c = canvas.Canvas(str(out), pagesize=A4)
    c.setTitle(TITLE)
    c.setAuthor("ozetle örnek belgesi (kurgusal)")
    page = 1
    y = height - margin

    def new_page() -> None:
        nonlocal page, y
        c.setFont("DejaVu", 8)
        c.drawString(margin, 28, f"Sayfa {page}")
        c.showPage()
        page += 1
        y = height - margin
        c.setFont("DejaVu", 8)
        c.drawString(margin, y, HEADER)
        y -= 2 * line_h

    def line(text: str, font_size: int = size) -> None:
        nonlocal y
        if y < margin + line_h:
            new_page()
        c.setFont("DejaVu", font_size)
        c.drawString(margin, y, text)
        y -= line_h if font_size <= size else line_h + 6

    c.setFont("DejaVu", 8)
    c.drawString(margin, y, HEADER)
    y -= 2 * line_h
    for title_line in textwrap.wrap(TITLE, 60):
        line(title_line, 15)
    y -= line_h
    for heading, body in SECTIONS:
        line(heading, 12)
        for text_line in textwrap.wrap(body, 68):
            line(text_line)
        y -= line_h // 2
    line("Kaynaklar", 12)
    for ref in REFERENCES:
        for text_line in textwrap.wrap(ref, 68):
            line(text_line)
    c.setFont("DejaVu", 8)
    c.drawString(margin, 28, f"Sayfa {page}")
    c.showPage()
    c.save()
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "ornek-makale.pdf")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(build(target))
