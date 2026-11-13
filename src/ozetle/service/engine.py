"""Ayarlardan model motorunu kurar (local ve Lambda worker ortak)."""

from __future__ import annotations

import functools

from ozetle.models.keywords import KeyBERTExtractor
from ozetle.models.summarizer import HFSummarizer, get_spec
from ozetle.service.config import Settings
from ozetle.service.processing import Engine


def build_engine(settings: Settings) -> Engine:
    """Motoru kurar; modeller ilk kullanıma kadar yüklenmez."""
    return Engine(
        summarizer=HFSummarizer(get_spec(settings.model)),
        keywords=KeyBERTExtractor(settings.keyword_model) if settings.keywords_enabled else None,
    )


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
