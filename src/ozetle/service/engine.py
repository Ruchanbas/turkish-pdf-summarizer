"""Ayarlardan model motorunu kurar (local ve Lambda worker ortak)."""

from __future__ import annotations

import functools
import importlib.util
import logging

from ozetle.models.keywords import KeyBERTExtractor, KeywordExtractor
from ozetle.models.summarizer import LEAD_MODEL, HFSummarizer, LeadSummarizer, Summarizer, get_spec
from ozetle.service.config import Settings
from ozetle.service.processing import Engine

logger = logging.getLogger(__name__)


def build_engine(settings: Settings) -> Engine:
    """Motoru kurar; modeller ilk kullanıma kadar yüklenmez.

    ``OZETLE_MODEL=lead`` ile model indirmeden çalışır. KeyBERT kurulu
    değilse anahtar kelimeler kapatılır ve bu loglanır.
    """
    summarizer: Summarizer = (
        LeadSummarizer() if settings.model == LEAD_MODEL else HFSummarizer(get_spec(settings.model))
    )
    keywords: KeywordExtractor | None = None
    if settings.keywords_enabled:
        if importlib.util.find_spec("keybert") is None:
            logger.warning("keybert kurulu değil; anahtar kelimeler kapalı.")
        else:
            keywords = KeyBERTExtractor(settings.keyword_model)
    return Engine(summarizer=summarizer, keywords=keywords)


def cached_engine_factory(settings: Settings) -> functools.partial[Engine]:
    """Süreç başına tek motor: Lambda konteyneri ve local sunucu modeli bir kez yükler."""
    return functools.partial(
        _engine_for, settings.model, settings.keyword_model, settings.keywords_enabled
    )


@functools.cache
def _engine_for(model: str, keyword_model: str, keywords_enabled: bool) -> Engine:
    return build_engine(
        Settings(model=model, keyword_model=keyword_model, keywords_enabled=keywords_enabled)
    )
