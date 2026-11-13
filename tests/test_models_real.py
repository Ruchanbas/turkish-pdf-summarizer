"""Gerçek Hugging Face modelleriyle duman testi.

Model indirir (yüzlerce MB), bu yüzden varsayılan test koşusunda çalışmaz:

    pytest -m model                       # varsayılan model
    OZETLE_MODEL=mt5-base pytest -m model # katalogdan başka bir model

CI'da ``models.yml`` workflow'u çalıştırır.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ozetle.models.keywords import TURKISH_STOPWORDS, KeyBERTExtractor
from ozetle.models.summarizer import DEFAULT_MODEL, HFSummarizer, get_spec, summarize_document
from ozetle.pipeline import prepare
from ozetle.text.clean import turkish_lower
from pdf_fixtures import make_article_pdf

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def summarizer() -> HFSummarizer:
    pytest.importorskip("transformers")
    s = HFSummarizer(get_spec(os.environ.get("OZETLE_MODEL", DEFAULT_MODEL)))
    s.warm_up()  # eksik ağırlık varsa burada ModelLoadError
    return s


def test_turkce_ozet(summarizer: HFSummarizer, tmp_path: Path) -> None:
    pdf = make_article_pdf(tmp_path / "m.pdf", n_body_pages=4)
    budget = summarizer.max_input_tokens - 16
    doc = prepare(pdf, max_tokens=budget, token_counter=summarizer.count_tokens)
    result = summarize_document([c.text for c in doc.chunks], summarizer)
    print(f"\n[{summarizer.name}] {result.rounds} tur, {result.seconds:.1f} sn\n{result.summary}")
    assert len(result.summary.split()) >= 5
    assert any(ch in result.summary for ch in "çğıöşüÇĞİÖŞÜ")


def test_anahtar_kelimeler(tmp_path: Path) -> None:
    pytest.importorskip("keybert")
    pdf = make_article_pdf(tmp_path / "m.pdf", n_body_pages=2)
    keywords = KeyBERTExtractor().extract(prepare(pdf).text, top_n=6)
    print("\n" + ", ".join(f"{k.phrase} ({k.score})" for k in keywords))
    assert keywords
    assert not any(turkish_lower(k.phrase) in TURKISH_STOPWORDS for k in keywords)
