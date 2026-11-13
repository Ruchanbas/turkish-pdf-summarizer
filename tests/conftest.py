"""Testler arasında paylaşılan sahte modeller ve fixture'lar."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from ozetle.models.keywords import Keyword
from pdf_fixtures import make_article_pdf


class FakeSummarizer:
    """Her metnin ilk ``keep`` kelimesini 'özet' olarak döndürür."""

    name = "fake/summarizer"

    def __init__(self, *, max_input_tokens: int = 120, keep: int = 8) -> None:
        self.max_input_tokens = max_input_tokens
        self.keep = keep
        self.calls: list[list[str]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def summarize_batch(self, texts: Sequence[str]) -> list[str]:
        self.calls.append(list(texts))
        return [" ".join(t.split()[: self.keep]) + "." for t in texts]


class FakeKeywords:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, text: str, *, top_n: int = 8) -> list[Keyword]:
        self.calls += 1
        return [Keyword("fizyoterapi", 0.71), Keyword("fonksiyonel skor", 0.64)][:top_n]


@pytest.fixture(scope="session")
def article_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path: Path = make_article_pdf(tmp_path_factory.mktemp("pdf") / "makale.pdf", n_body_pages=3)
    return path.read_bytes()


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """moto için sahte kimlik bilgileri; gerçek hesaba asla gidilmez."""
    for key, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "eu-central-1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
