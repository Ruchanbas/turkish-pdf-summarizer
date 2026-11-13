"""Gerçek transformers koduyla, ağ gerektirmeden model testleri.

Yerelde küçük bir Unigram (SentencePiece tarzı) tokenizer eğitilir ve rastgele ağırlıklı
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
tokenizers = pytest.importorskip("tokenizers")
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
def tiny_tokenizer(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    special = ["<pad>", "</s>", "<unk>"]
    sp = tokenizers.SentencePieceUnigramTokenizer()
    sp.train_from_iterator(
        [finding(i) for i in range(1, 200)],
        vocab_size=300,
        special_tokens=special,
        unk_token="<unk>",
        show_progress=False,
    )
    # transformers 4 ve 5'te aynı biçimde çalışan yol: tokenizers nesnesini sarmala.
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=sp._tokenizer,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    tokenizer._tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", tokenizer.eos_token_id)]
    )
    return tokenizer


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory, tiny_tokenizer) -> Path:  # type: ignore[no-untyped-def]
    root = tmp_path_factory.mktemp("tiny")
    tokenizer = tiny_tokenizer
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
    return ModelSpec(
        key="tiny",
        model_id=str(path),
        max_new_tokens=8,
        min_new_tokens=4,  # rastgele model aksi halde hemen </s> üretir
        num_beams=2,
        **kw,  # type: ignore[arg-type]
    )


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
    assert all(isinstance(s, str) and s for s in out)
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
    assert result.summary


# ---------------------------------------------------------------------------
# Paylaşımlı encoder-decoder (bert2bert "shared") — transformers 5 regresyonu
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def legacy_shared(tmp_path_factory: pytest.TempPathFactory, tiny_tokenizer) -> Path:  # type: ignore[no-untyped-def]
    """transformers 4 döneminin bert2bert_shared kaydını birebir taklit eder.

    ``tie_encoder_decoder=True`` ile kaydedilmiş modelde decoder'ın encoder
    ile ortak katmanları kayıtta yoktur; yalnızca çapraz dikkat ve çıkış
    katmanı vardır.
    """
    tok = tiny_tokenizer
    common = dict(
        vocab_size=len(tok), hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64,
    )  # fmt: skip
    config = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(
        transformers.BertConfig(**common),
        transformers.BertConfig(**common, is_decoder=True, add_cross_attention=True),
        tie_encoder_decoder=True,
    )
    config.decoder_start_token_id = tok.pad_token_id
    config.pad_token_id = tok.pad_token_id
    config.eos_token_id = tok.eos_token_id
    torch.manual_seed(0)
    state = transformers.EncoderDecoderModel(config=config).state_dict()
    encoder_names = {k.removeprefix("encoder.") for k in state if k.startswith("encoder.")}
    legacy = {
        k: v.clone().contiguous()
        for k, v in state.items()
        if not (k.startswith("decoder.bert.") and k.removeprefix("decoder.bert.") in encoder_names)
    }
    out = tmp_path_factory.mktemp("legacy") / "bert2bert"
    config.save_pretrained(out)
    torch.save(legacy, out / "pytorch_model.bin")
    tok.save_pretrained(out)
    return out


def test_paylasimli_kayit_bagli_yuklenir(legacy_shared: Path) -> None:
    """Gerçek bert2bert'te bulundu: transformers 5 bu kayıtta decoder'ı
    encoder'a bağlamıyor, 198 katmanı rastgele başlatıyor. Katı yükleyici
    alarm verdi; yükleyici bağı açıkça kuruyor ve bellekte kanıtlıyor."""
    _, model = load_seq2seq(str(legacy_shared))
    enc = model.encoder.embeddings.word_embeddings.weight
    dec = model.decoder.bert.embeddings.word_embeddings.weight
    assert dec.data_ptr() == enc.data_ptr()  # kopya değil, aynı bellek
    q_enc = model.encoder.encoder.layer[0].attention.self.query.weight
    q_dec = model.decoder.bert.encoder.layer[0].attention.self.query.weight
    assert q_dec is q_enc
    head = model.decoder.get_output_embeddings().weight
    assert head.data_ptr() == enc.data_ptr()  # çıkış katmanı da aynı gömmede


def test_paylasimli_model_ozet_uretir(legacy_shared: Path) -> None:
    spec = ModelSpec(key="b2b", model_id=str(legacy_shared), max_new_tokens=6, min_new_tokens=2)
    out = HFSummarizer(spec).summarize_batch([finding(1)])
    assert len(out) == 1


def test_paylasimli_kayitta_gercek_eksik_yakalanir(legacy_shared: Path, tmp_path: Path) -> None:
    """Bağ kurma yalnızca encoder'da karşılığı olan katmanlara uygulanır.
    Çapraz dikkat gibi decoder'a özgü bir katman eksikse yine hatadır."""
    broken = tmp_path / "broken"
    shutil.copytree(legacy_shared, broken)
    state = torch.load(broken / "pytorch_model.bin", weights_only=True)
    victim = next(k for k in state if "crossattention" in k)
    del state[victim]
    torch.save(state, broken / "pytorch_model.bin")
    with pytest.raises(ModelLoadError, match="rastgele"):
        load_seq2seq(str(broken))
