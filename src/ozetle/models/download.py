"""Modelleri önceden indirir ve katı yüklemeyle doğrular.

Docker imajı derlenirken çalışır: Lambda soğuk başlangıcında ağdan model
indirilmez, imaj içinden okunur. Yükleme burada bir kez denenir; eksik
ağırlık varsa (bkz. :func:`~ozetle.models.loader.check_loading_info`) imaj
derlemesi başarısız olur ve bozuk model asla canlıya çıkmaz.

    python -m ozetle.models.download --model mt5-small
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ozetle.models.keywords import DEFAULT_EMBEDDING_MODEL, KeyBERTExtractor
from ozetle.models.summarizer import DEFAULT_MODEL, LEAD_MODEL, HFSummarizer, get_spec

#: Çıkarımda kullanılmayan çerçevelerin ağırlıkları indirilmez.
IGNORE = ["*.h5", "*.msgpack", "*.onnx", "*.ot", "*.tflite", "onnx/*", "openvino/*", "*.gguf"]


def fetch(repo_id: str) -> str:
    from huggingface_hub import HfApi, snapshot_download

    files = HfApi().list_repo_files(repo_id)
    ignore = list(IGNORE)
    # Aynı ağırlığın hem safetensors hem .bin kopyası varsa yalnızca biri.
    if any(f.endswith(".safetensors") for f in files):
        ignore.append("*.bin")
    path: str = snapshot_download(repo_id, ignore_patterns=ignore)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ozetle.models.download")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--keyword-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--no-keywords", action="store_true")
    args = parser.parse_args(argv)

    if args.model != LEAD_MODEL:
        spec = get_spec(args.model)
        print(f"indiriliyor: {spec.model_id}")
        fetch(spec.model_id)
        summarizer = HFSummarizer(spec)
        summarizer.warm_up()  # eksik ağırlık → ModelLoadError → derleme durur
        sample = summarizer.summarize_batch(
            ["Ankara'da yapılan çalışmada hastaların fonksiyonel skorları anlamlı ölçüde arttı."]
        )
        print(f"doğrulandı ({summarizer.load_seconds:.1f} sn): {sample[0]!r}")

    if not args.no_keywords:
        print(f"indiriliyor: {args.keyword_model}")
        fetch(args.keyword_model)
        keywords = KeyBERTExtractor(args.keyword_model).extract(
            "Fizyoterapi programı diz osteoartriti hastalarında ağrıyı azalttı.", top_n=3
        )
        print("doğrulandı:", [k.phrase for k in keywords])


if __name__ == "__main__":  # pragma: no cover
    main()
