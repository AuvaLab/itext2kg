#!/usr/bin/env python3
"""
Score SOTA detectors with replicates for stochastic methods.

Yoon and BEAM are deterministic (n=1). BERTrend is run RUN_TIMES times with
different UMAP seeds. C-Unseen reuses the RUN_TIMES detection files already
produced by ablation_c_unseen.py.

Usage:
    pyenv activate venv-itext2kg-update
    python evaluation/c_unseen/run/run_sota_replicates.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

for _p in Path(__file__).resolve().parents:
    if (_p / ".git").exists():
        project_root = _p
        break
else:
    raise RuntimeError("Could not locate repository root")
sys.path.insert(0, str(project_root))

from evaluation.c_unseen.paths import DETECTORS, OUTPUTS, SCORING  # noqa: E402

# ==========================
# User-configurable globals
# ==========================
OUTPUT_DIR: Path = OUTPUTS
ABLATION_DIR: Path = OUTPUT_DIR / "ablation"
REPLICATES_DIR: Path = OUTPUT_DIR / "sota_replicates"

RUN_TIMES: int = 5
BERTREND_SEEDS: list[int] = [42, 43, 44, 45, 46]

SCALAR_METRIC_KEYS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "detections_total",
    "true_positives",
    "events_covered",
    "mean_lead_days",
    "mean_lead_years",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(var)


def fmt_ms(mean: float, std: float, digits: int = 3) -> str:
    if std == 0.0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_scorer():
    return load_module("weak_signal_scorer", SCORING / "scorer.py")


def load_bertrend():
    return load_module(
        "bertrend_topic_modeling",
        DETECTORS / "bertrend_topic_modeling.py",
    )


def run_bertrend_once(bt, slices, embeddings_by_year, embeddings_model, seed: int) -> list[dict]:
    bt.RANDOM_STATE = seed
    np.random.seed(seed)

    embedding_cache: dict[str, np.ndarray] = {}

    def embed_text(text: str) -> np.ndarray:
        if text not in embedding_cache:
            vec = embeddings_model.embed_documents([text])[0]
            embedding_cache[text] = np.asarray(vec, dtype=np.float64)
        return embedding_cache[text]

    global_topics = []
    next_topic_id = 0
    all_years = [year for year, _ in slices]
    for year, texts in slices:
        slice_topics = bt.fit_slice_topics(year, texts, embeddings_by_year[year])
        n_merged, n_new, next_topic_id = bt.merge_topics(
            global_topics,
            slice_topics,
            year,
            embed_text,
            next_topic_id,
        )
        del n_merged, n_new
        bt.update_popularity(global_topics, year)
        labels = bt.classify_signals(global_topics, year, all_years)
        for topic in global_topics:
            if topic.topic_id in labels:
                topic.label_by_year[year] = labels[topic.topic_id]
    return bt.extract_detections(global_topics)


def score_path(scorer, method: str, path: Path, signals_meta, anchors, overrides) -> dict:
    return scorer.score_file(method, path, signals_meta, anchors, overrides)


def print_tables(summary: dict[str, dict], k_levels: list[int]) -> None:
    header = (
        f"{'method':<12} | {'n':>3} | {'detections':>14} | {'precision':>13} | "
        f"{'recall':>13} | {'f1':>13} | {'events':>10} | {'lead_yrs':>12}"
    )
    for k in k_levels:
        print(f"\n--- k = {k}  (mean ± sample std) ---")
        print(header)
        print("-" * len(header))
        for method in ("yoon", "bertrend", "beam", "c_unseen"):
            if method not in summary:
                continue
            row = summary[method][f"k={k}"]
            n = summary[method]["n"]
            print(
                f"{method:<12} | {n:>3} | "
                f"{fmt_ms(row['detections_total']['mean'], row['detections_total']['std'], 1):>14} | "
                f"{fmt_ms(row['precision']['mean'], row['precision']['std']):>13} | "
                f"{fmt_ms(row['recall']['mean'], row['recall']['std']):>13} | "
                f"{fmt_ms(row['f1']['mean'], row['f1']['std']):>13} | "
                f"{fmt_ms(row['events_covered']['mean'], row['events_covered']['std'], 2):>10} | "
                f"{fmt_ms(row['mean_lead_years']['mean'], row['mean_lead_years']['std'], 2):>12}"
            )


def aggregate(runs_by_method: dict[str, list[dict]], k_levels: list[int]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for method, runs in runs_by_method.items():
        per_k: dict[str, dict] = {"n": len(runs)}
        for k in k_levels:
            key = f"k={k}"
            metrics: dict[str, dict] = {}
            for metric in SCALAR_METRIC_KEYS:
                values = [float(run[key][metric]) for run in runs]
                mean, std = mean_std(values)
                metrics[metric] = {"mean": mean, "std": std, "values": values}
            per_k[key] = metrics
        summary[method] = per_k
    return summary


def main() -> None:
    if len(BERTREND_SEEDS) != RUN_TIMES:
        raise ValueError("BERTREND_SEEDS must have length RUN_TIMES")

    scorer = load_scorer()
    with open(scorer.BENCHMARK_JSON, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    signals_meta = scorer.compute_windows(benchmark)
    anchors = scorer.load_anchors(scorer.ANCHORS_JSON)
    overrides = scorer.load_overrides(scorer.OVERRIDES_JSON)

    runs_by_method: dict[str, list[dict]] = {}

    for method, filename in (("yoon", "detections_yoon.json"), ("beam", "detections_beam.json")):
        path = OUTPUT_DIR / filename
        print(f"Scoring deterministic {method}: {path.name}")
        runs_by_method[method] = [
            score_path(scorer, method, path, signals_meta, anchors, overrides)
        ]

    print(f"\nRunning BERTrend {RUN_TIMES} times (seeds={BERTREND_SEEDS})")
    bt = load_bertrend()
    facts = bt.load_data(bt.INPUT_JSON)
    slices = bt.build_year_slices(facts)
    api_key = bt._resolve_openai_api_key()
    from langchain_openai import OpenAIEmbeddings

    embeddings_model = OpenAIEmbeddings(api_key=api_key, model=bt.EMBEDDING_MODEL)
    print("Embedding facts once with", bt.EMBEDDING_MODEL)
    embeddings_by_year = bt.embed_all_facts(slices, embeddings_model)

    bertrend_runs: list[dict] = []
    for run_id, seed in enumerate(BERTREND_SEEDS):
        print(f"  BERTrend run={run_id} seed={seed}")
        detections = run_bertrend_once(bt, slices, embeddings_by_year, embeddings_model, seed)
        path = REPLICATES_DIR / f"detections_bertrend_run{run_id}.json"
        bt.save(detections, path)
        print(f"    {len(detections)} detections -> {path.name}")
        scored = score_path(scorer, "bertrend", path, signals_meta, anchors, overrides)
        save_json(scored, REPLICATES_DIR / f"results_bertrend_run{run_id}.json")
        bertrend_runs.append(scored)
        print(f"    f1@k2={scored['k=2']['f1']:.3f}  f1@k3={scored['k=3']['f1']:.3f}")
    runs_by_method["bertrend"] = bertrend_runs

    print(f"\nScoring C-Unseen {RUN_TIMES} existing ablation replicates")
    cunseen_runs: list[dict] = []
    for run_id in range(RUN_TIMES):
        path = ABLATION_DIR / f"detections_c_unseen_run{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run evaluation/c_unseen/run/ablation_c_unseen.py first."
            )
        print(f"  C-Unseen run={run_id}: {path.name}")
        scored = score_path(scorer, "c_unseen", path, signals_meta, anchors, overrides)
        save_json(scored, REPLICATES_DIR / f"results_c_unseen_run{run_id}.json")
        cunseen_runs.append(scored)
        print(f"    f1@k2={scored['k=2']['f1']:.3f}  f1@k3={scored['k=3']['f1']:.3f}")
    runs_by_method["c_unseen"] = cunseen_runs

    summary = aggregate(runs_by_method, scorer.K_LEVELS)
    print_tables(summary, scorer.K_LEVELS)

    payload = {
        "run_times": RUN_TIMES,
        "bertrend_seeds": BERTREND_SEEDS,
        "notes": {
            "yoon": "deterministic, n=1",
            "beam": "deterministic, n=1",
            "bertrend": "UMAP random_state varies per run; embeddings computed once",
            "c_unseen": "reuses ablation_c_unseen.py replicates",
        },
        "scorer": summary,
        "per_run": {
            method: [
                {
                    f"k={k}": {metric: run[f"k={k}"][metric] for metric in SCALAR_METRIC_KEYS}
                    for k in scorer.K_LEVELS
                }
                for run in runs
            ]
            for method, runs in runs_by_method.items()
        },
    }
    out = REPLICATES_DIR / "results_sota_summary.json"
    save_json(payload, out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    print("=" * 50)
    print("  SOTA REPLICATES: YOON/BEAM n=1, BERTREND/C-UNSEEN n=5")
    print("=" * 50)
    main()
