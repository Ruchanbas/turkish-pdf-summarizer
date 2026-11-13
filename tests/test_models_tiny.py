"""Gerçek transformers koduyla, ağ gerektirmeden model testleri.

Yerelde küçük bir SentencePiece tokenizer eğitilir ve rastgele ağırlıklı
minik bir T5 kaydedilir. Çıktı anlamsızdır ama yükleme, katı ağırlık
denetimi, üretim ve map-reduce gerçek kütüphane üzerinden çalışır.

torch/transformers kurulu değilse atlanır (CI'da ayrı bir işte çalışır).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
spm = pytest.importorskip("sentencepiece")
safetensors_torch = pytest.importorskip("safetensors.torch")

from ozetle.models.loader import ModelLoadError  # noqa: E402
from ozetle.models.summarizer import (  # noqa: E402
    HFSummarizer,
    ModelSpec,
    load_seq2seq,
    summarize_document,
)
from ozetle.pipeline import prepare  # noqa: E402
from pdf_fixtures import finding, make_article_pdf  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tiny")
    corpus = root / "corpus.txt"
    corpus.write_text("\n".join(finding(i) for i in range(1, 200)), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(root / "spiece"),
        vocab_size=300,
        character_coverage=1.0,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        minloglevel=2,
    )
    tokenizer = transformers.T5Tokenizer(vocab_file=str(root / "spiece.model"), extra_ids=0)
    config = transformers.T5Config(
        vocab_size=len(tokenizer),
        d_model=32,
        d_ff=64,
        d_kv=8,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(0)
    model = transformers.T5ForConditionalGeneration(config)
    out = root / "model"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


def _spec(path: Path, **kw: object) -> ModelSpec:
    return ModelSpec(key="tiny", model_id=str(path), max_new_tokens=8, num_beams=2, **kw)  # type: ignore[arg-type]


def test_katı_yukleme_basarili(tiny_model: Path) -> None:
    tokenizer, model = load_seq2seq(str(tiny_model))
    assert not model.training  # eval moduna alınmış
    assert len(tokenizer("Çalışma yapılmıştır.")["input_ids"]) > 1


def test_eksik_agirlik_yakalanir(tiny_model: Path, tmp_path: Path) -> None:
    """strict=False regresyonu, gerçek transformers ile.

    Bir katman kayıttan silinir. transformers bu katmanı sessizce rastgele
    başlatır; load_seq2seq ise yüklemeyi reddetmelidir.
    """
    broken = tmp_path / "broken"
    shutil.copytree(tiny_model, broken)
    weights = broken / "model.safetensors"
    state = safetensors_torch.load_file(str(weights))
    victim = next(k for k in state if k.startswith("encoder.block.0.layer.0.SelfAttention.q"))
    del state[victim]
    safetensors_torch.save_file(state, str(weights), metadata={"format": "pt"})

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(broken)  # sessizce yüklenir
    assert model is not None
    with pytest.raises(ModelLoadError, match="rastgele"):
        load_seq2seq(str(broken))


def test_ozet_uretir(tiny_model: Path) -> None:
    summarizer = HFSummarizer(_spec(tiny_model), batch_size=2)
    out = summarizer.summarize_batch([finding(1), finding(2), finding(3)])
    assert len(out) == 3
    assert all(isinstance(s, str) for s in out)
    assert summarizer.count_tokens(finding(1)) > 10


def test_uctan_uca_map_reduce(tiny_model: Path, tmp_path: Path) -> None:
    summarizer = HFSummarizer(_spec(tiny_model, max_input_tokens=128))
    pdf = make_article_pdf(tmp_path / "m.pdf", n_body_pages=3)
    doc = prepare(pdf, max_tokens=112, token_counter=summarizer.count_tokens)
    assert len(doc.chunks) > 1
    assert all(summarizer.count_tokens(c.text) <= 128 for c in doc.chunks)
    result = summarize_document([c.text for c in doc.chunks], summarizer)
    assert len(result.partials) == len(doc.chunks)
    assert result.rounds >= 2
