"""Özetleme modellerinin karşılaştırması (Faz 2).

Her model **üretimdeki kod yolundan** geçer: aynı parçalayıcı, aynı
map-reduce, aynı üretim ayarları. Böylece tablo, servisin gerçekte
vereceği kaliteyi ve gecikmeyi ölçer, modelin kağıt üzerindeki halini
değil.

Kullanım::

    python bench/run.py probe                     # modeller ve veri seti iniyor mu
    python bench/run.py run --model mt5-small --n 100 --out results/
    python bench/run.py report results/           # Markdown tablo

Metrikler:
    ROUGE-1/2/L  Türkçe farkında tokenizer ile (aşağıya bakın)
    BERTScore F1 çok dilli BERT ile anlamsal benzerlik
    Gecikme      makale başına p50 / p95, CPU, tek iş parçacığı değil
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from ozetle.models.summarizer import (
    LEAD_MODEL,
    MODELS,
    HFSummarizer,
    LeadSummarizer,
    get_spec,
    summarize_document,
)
from ozetle.text.chunk import chunk_text
from ozetle.text.clean import turkish_lower

SEED = 42


# ---------------------------------------------------------------------------
# ROUGE için Türkçe tokenizer
# ---------------------------------------------------------------------------


class TurkishTokenizer:
    """rouge-score'un varsayılan tokenizer'ı Türkçe için kullanılamaz.

    Varsayılan tokenizer küçük harfe çevirip ``[^a-z0-9]`` dışındaki her şeyi
    boşluğa çevirir. "Çalışmasında İSTANBUL" → ``al mas nda i stanbul``: Türkçe harfler silinir,
    kelimeler parçalanır ve anlamsız parçalar eşleşir. Skorlar hem yanlış hem
    modeller arasında karşılaştırılamaz olur. Burada Unicode kelime sınırı ve
    Türkçe küçültme kullanılır.
    """

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", turkish_lower(text))


# ---------------------------------------------------------------------------
# Veri seti
# ---------------------------------------------------------------------------

#: Denenecek kaynaklar, sırayla. MLSUM yükleme betiğine dayanır; datasets 4
#: betikleri çalıştırmaz, bu yüzden Hugging Face'in parquet dönüşümü de denenir.
SOURCES: list[tuple[str, dict[str, Any]]] = [
    (
        "reciTAL/mlsum (tu, parquet)",
        {
            "path": "parquet",
            "data_files": {
                "test": "hf://datasets/reciTAL/mlsum@refs/convert/parquet/tu/test/*.parquet"
            },
        },
    ),
    ("reciTAL/mlsum (tu)", {"path": "reciTAL/mlsum", "name": "tu"}),
    ("mlsum (tu)", {"path": "mlsum", "name": "tu"}),
]


def load_examples(n: int) -> tuple[str, list[dict[str, str]]]:
    from datasets import load_dataset

    errors = []
    for label, kwargs in SOURCES:
        try:
            ds = load_dataset(split="test", **kwargs)
        except Exception as exc:  # her kaynağın hatası farklı
            errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        rows = [{"text": r["text"], "summary": r["summary"]} for r in ds]
        random.Random(SEED).shuffle(rows)
        return label, rows[:n]
    raise RuntimeError("Hiçbir veri kaynağı yüklenemedi:\n" + "\n".join(errors))


# ---------------------------------------------------------------------------
# Komutlar
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download, model_info

    for spec in MODELS.values():
        print(f"\n=== {spec.key}: {spec.model_id}")
        try:
            info = model_info(spec.model_id)
            params = info.safetensors.total if info.safetensors else None
            license_ = info.card_data.get("license") if info.card_data else None
            print(f"pipeline={info.pipeline_tag} params={params} license={license_}")
            config = json.loads(Path(hf_hub_download(spec.model_id, "config.json")).read_text())
            print(
                "architectures:",
                config.get("architectures"),
                "model_type:",
                config.get("model_type"),
            )
            readme = Path(hf_hub_download(spec.model_id, "README.md")).read_text(errors="ignore")
            print("--- README (ilk 80 satır) ---")
            print("\n".join(readme.splitlines()[:80]))
        except Exception as exc:
            print(f"HATA: {type(exc).__name__}: {exc}")

    print("\n=== Veri seti")
    label, rows = load_examples(3)
    print("kaynak:", label)
    for row in rows:
        print("-" * 40)
        print("METİN:", row["text"][:300].replace("\n", " "))
        print("ÖZET :", row["summary"][:200])


def cmd_prefix(args: argparse.Namespace) -> None:
    """Model kartı olmayan modeller için görev önekini ölçerek seç.

    Aynı örnekler her önekle özetlenir; ROUGE-L en yüksek olan önek
    katalogdaki ``ModelSpec.prefix`` değeri olur.
    """
    from dataclasses import replace

    from rouge_score import rouge_scorer

    _, rows = load_examples(args.n)
    scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=TurkishTokenizer())
    base = get_spec(args.model)
    shared = HFSummarizer(base)
    shared.warm_up()
    bundle = shared._bundle.get()
    for prefix in args.prefixes:
        s = HFSummarizer(replace(base, prefix=prefix), loader=lambda _id: bundle)
        outputs = s.summarize_batch([r["text"] for r in rows])
        score = statistics.fmean(
            scorer.score(r["summary"], o)["rougeL"].fmeasure
            for r, o in zip(rows, outputs, strict=True)
        )
        print(f"önek={prefix!r:12} ROUGE-L={100 * score:.2f}  örnek: {outputs[0][:160]}")


def _summarizer(key: str) -> Any:
    if key == LEAD_MODEL:
        return LeadSummarizer()
    return HFSummarizer(get_spec(key))


def cmd_run(args: argparse.Namespace) -> None:
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer

    source, rows = load_examples(args.n)
    summarizer = _summarizer(args.model)

    load_seconds = 0.0
    params = None
    if isinstance(summarizer, HFSummarizer):
        summarizer.warm_up()
        load_seconds = summarizer.load_seconds or 0.0
        params = summarizer._bundle.get()[1].num_parameters()

    candidates: list[str] = []
    latencies: list[float] = []
    chunks_per_doc: list[int] = []
    budget = summarizer.max_input_tokens - 16
    for i, row in enumerate(rows):
        started = time.perf_counter()
        chunks = chunk_text(row["text"], max_tokens=budget, token_counter=summarizer.count_tokens)
        result = summarize_document([c.text for c in chunks], summarizer)
        latencies.append(time.perf_counter() - started)
        candidates.append(result.summary)
        chunks_per_doc.append(len(chunks))
        if i % 10 == 0:
            print(f"[{args.model}] {i + 1}/{len(rows)} {latencies[-1]:.1f} sn", flush=True)

    references = [r["summary"] for r in rows]
    types = ["rouge1", "rouge2", "rougeL"]
    turkish = rouge_scorer.RougeScorer(types, tokenizer=TurkishTokenizer())
    naive = rouge_scorer.RougeScorer(["rouge1"])
    scores = [turkish.score(ref, cand) for ref, cand in zip(references, candidates, strict=True)]
    naive_r1 = [
        naive.score(ref, cand)["rouge1"].fmeasure
        for ref, cand in zip(references, candidates, strict=True)
    ]
    _, _, f1 = bert_score(candidates, references, lang="tr", batch_size=16, verbose=False)

    def mean(values: list[float]) -> float:
        return round(100 * statistics.fmean(values), 2)

    latencies_sorted = sorted(latencies)
    result = {
        "model": args.model,
        "model_id": getattr(summarizer, "name", args.model),
        "dataset": source,
        "n": len(rows),
        "rouge1": mean([s["rouge1"].fmeasure for s in scores]),
        "rouge2": mean([s["rouge2"].fmeasure for s in scores]),
        "rougeL": mean([s["rougeL"].fmeasure for s in scores]),
        "rouge1_default_tokenizer": mean(naive_r1),
        "bertscore_f1": mean([float(x) for x in f1]),
        "latency_p50": round(statistics.median(latencies), 2),
        "latency_p95": round(latencies_sorted[int(0.95 * (len(latencies) - 1))], 2),
        "chunks_mean": round(statistics.fmean(chunks_per_doc), 2),
        "summary_words_mean": round(statistics.fmean(len(c.split()) for c in candidates), 1),
        "load_seconds": round(load_seconds, 1),
        "params": params,
        "machine": f"{platform.machine()} {platform.processor() or ''}".strip(),
        "samples": [
            {"reference": references[i], "candidate": candidates[i]}
            for i in range(min(3, len(rows)))
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.model}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "samples"}, ensure_ascii=False, indent=2
        )
    )


def cmd_report(args: argparse.Namespace) -> None:
    results = [json.loads(p.read_text()) for p in sorted(Path(args.dir).glob("*.json"))]
    results.sort(key=lambda r: r["rougeL"], reverse=True)
    if not results:
        sys.exit("Sonuç yok")
    lines = [
        f"MLSUM-TR test, n={results[0]['n']} (seed {SEED}), {results[0]['dataset']}",
        "",
        "| Model | Parametre | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 "
        "| p50 (sn) | p95 (sn) | Yükleme (sn) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        params = f"{r['params'] / 1e6:.0f}M" if r["params"] else "-"
        lines.append(
            f"| {r['model']} | {params} | {r['rouge1']} | {r['rouge2']} | {r['rougeL']} | "
            f"{r['bertscore_f1']} | {r['latency_p50']} | {r['latency_p95']} | {r['load_seconds']} |"
        )
    lines += [
        "",
        "Varsayılan rouge-score tokenizer'ı ile ROUGE-1 (Türkçe harfleri siler):",
        "",
        *(
            f"- {r['model']}: {r['rouge1_default_tokenizer']} (Türkçe tokenizer ile {r['rouge1']})"
            for r in results
        ),
    ]
    report = "\n".join(lines)
    print(report)
    (Path(args.dir) / "report.md").write_text(report + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    prefix = sub.add_parser("prefix")
    prefix.add_argument("--model", required=True, choices=list(MODELS))
    prefix.add_argument("--n", type=int, default=8)
    prefix.add_argument("--prefixes", nargs="+", default=["", "[S2S] ", "[S2S]"])
    run = sub.add_parser("run")
    run.add_argument("--model", required=True, choices=[LEAD_MODEL, *MODELS])
    run.add_argument("--n", type=int, default=100)
    run.add_argument("--out", default="results")
    report = sub.add_parser("report")
    report.add_argument("dir")
    args = parser.parse_args()
    commands = {"probe": cmd_probe, "prefix": cmd_prefix, "run": cmd_run, "report": cmd_report}
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
