#!/usr/bin/env python3
"""
C-Unseen weak-signal detector for the Wiki-OpenAI benchmark.

Runs (or reuses) the whole-snapshot C-Unseen pipeline, extracts one detection
per weak-signal bucket from bridging-subgraph traces, and writes
detections_c_unseen.json for the shared scorer.

Usage:
    pyenv activate venv-itext2kg-update
    python evaluation/c_unseen/detectors/c_unseen_runner.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import pickle
import re
import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / ".git").exists():
        project_root = _p
        break
else:
    raise RuntimeError("Could not locate repository root")
sys.path.insert(0, str(project_root))

from api_keys import openai_api_key  # noqa: E402
from evaluation.c_unseen.paths import OUTPUTS, REPORTS, SNAPSHOTS  # noqa: E402

# ==========================
# User-configurable globals
# ==========================
SNAPSHOTS_DIR: Path = SNAPSHOTS
REPORTS_DIR: Path = REPORTS
OUTPUT_JSON: Path = OUTPUTS / "detections_c_unseen.json"

YEAR_START: str | None = "2015"
YEAR_END: str | None = "2025"
CENTRAL_ENTITY = "openai"

OPENAI_MODEL_NAME = "gpt-5.4-mini-2026-03-17"
OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-large"
TEMPERATURE = 0
MAX_TOKENS = None
TIMEOUT = None
MAX_RETRIES = 2

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _load_run_script():
    spec = importlib.util.spec_from_file_location(
        "run_c_unseen_whole_snapshot",
        project_root
        / "evaluation"
        / "c_unseen"
        / "run"
        / "run_c_unseen_whole_snapshot.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run_c_unseen_whole_snapshot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_run_script(runner_mod) -> None:
    runner_mod.SNAPSHOTS_DIR = SNAPSHOTS_DIR
    runner_mod.REPORTS_DIR = REPORTS_DIR
    runner_mod.YEAR_START = YEAR_START
    runner_mod.YEAR_END = YEAR_END
    runner_mod.CENTRAL_ENTITY = CENTRAL_ENTITY


def _trace_ok(runner_mod, label: str, kind: str) -> bool:
    path = REPORTS_DIR / f"{kind}_{label}.txt"
    return runner_mod._extract_response_json(path) is not None


def _traces_complete(runner_mod, labels: list[str]) -> bool:
    for idx, label in enumerate(labels):
        if not _trace_ok(runner_mod, label, "whole_snapshot_rare"):
            return False
        if idx > 0 and not _trace_ok(runner_mod, label, "weak_signal_bridging"):
            return False
    return True


def _hydrate_rare_flags(runner_mod, kg, label: str) -> None:
    data = runner_mod._extract_response_json(
        REPORTS_DIR / f"whole_snapshot_rare_{label}.txt"
    )
    if not data:
        return
    for idx in runner_mod._normalize_index_list(data.get("rare_indices")):
        if 0 <= idx < len(kg.relationships):
            kg.relationships[idx].properties.rare = True


def _triple_texts(bucket: dict) -> list[str]:
    return [triple for _, _, triple in bucket["triples"]]


def _build_description(bucket: dict) -> str:
    parts: list[str] = []
    for triple in _triple_texts(bucket):
        parts.extend(_TOKEN_RE.split(triple.lower()))
    theme = (bucket.get("theme") or "").strip()
    explanation = (bucket.get("explanation") or "").strip()
    if theme:
        parts.extend(_TOKEN_RE.split(theme.lower()))
    if explanation:
        parts.extend(_TOKEN_RE.split(explanation.lower()))
    tokens = [token for token in parts if token]
    return " ".join(tokens)


def _bucket_to_detection(runner_mod, kg, label: str, bucket: dict) -> dict:
    triples = _triple_texts(bucket)
    return {
        "year": int(label),
        "description": _build_description(bucket),
        "bucket_id": bucket["bucket_id"],
        "theme": bucket.get("theme", ""),
        "explanation": bucket.get("explanation", ""),
        "kg_indices": bucket.get("kg_indices", []),
        "bridging_indices": bucket.get("bridging_indices", []),
        "triples": triples,
    }


def _load_kgs(runner_mod, snapshot_paths: list[Path]):
    kgs = []
    for path in snapshot_paths:
        with open(path, "rb") as f:
            src_kg = runner_mod._unwrap_pickled_snapshot(pickle.load(f))
        if not hasattr(src_kg, "entities") or not hasattr(src_kg, "relationships"):
            raise TypeError(
                f"{path}: expected KnowledgeGraph-like object, got {type(src_kg).__name__}"
            )
        kgs.append(runner_mod._to_signal_kg(src_kg))
    return kgs


async def _maybe_run_c_unseen(
    runner_mod,
    kgs,
    labels: list[str],
    traces_complete: bool,
):
    if traces_complete:
        print("All required traces present — skipping LLM pipeline.")
        return kgs

    if not openai_api_key:
        raise RuntimeError(
            "Missing traces and openai_api_key is empty in api_keys.py"
        )

    os.environ["OPENAI_API_KEY"] = openai_api_key
    from langchain_openai import OpenAIEmbeddings
    from itext2kg.c_unseen import CUnseen

    print("Missing or unparseable traces — running C-Unseen pipeline...")
    llm_kwargs = {
        "api_key": openai_api_key,
        "model": OPENAI_MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "timeout": TIMEOUT,
        "max_retries": MAX_RETRIES,
    }
    if runner_mod._model_supports_temperature(OPENAI_MODEL_NAME):
        llm_kwargs["temperature"] = TEMPERATURE
    else:
        print(f"Model {OPENAI_MODEL_NAME!r} does not support temperature — omitting it.")
    llm_model = runner_mod._build_chat_openai(**llm_kwargs)
    embeddings_model = OpenAIEmbeddings(
        api_key=openai_api_key,
        model=OPENAI_EMBEDDINGS_MODEL,
    )
    cunseen = CUnseen(
        llm_model=llm_model,
        embeddings_model=embeddings_model,
        reports_dir=REPORTS_DIR,
    )
    return await cunseen.process_snapshots(
        kgs,
        snapshot_labels=labels,
        central_entity_name=CENTRAL_ENTITY,
    )


def _extract_detections(runner_mod, kgs, labels: list[str]) -> list[dict]:
    detections: list[dict] = []
    for kg, label in zip(kgs, labels):
        _hydrate_rare_flags(runner_mod, kg, label)
        buckets = runner_mod._weak_buckets_from_trace(kg, label)
        for bucket in buckets:
            detections.append(_bucket_to_detection(runner_mod, kg, label, bucket))
    detections.sort(key=lambda row: (row["year"], row["bucket_id"]))
    return detections


def save(detections: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)
        f.write("\n")


def _print_summary(detections: list[dict]) -> None:
    by_year: dict[int, int] = {}
    for det in detections:
        by_year[det["year"]] = by_year.get(det["year"], 0) + 1

    print("Weak-signal buckets per year:")
    for year in sorted(by_year):
        print(f"  {year}: {by_year[year]} bucket(s)")
    print(f"Total detections: {len(detections)}")


async def main_async() -> None:
    runner_mod = _load_run_script()
    _configure_run_script(runner_mod)

    snapshot_paths = runner_mod._get_snapshot_paths()
    labels = [runner_mod._snapshot_label_from_path(p) for p in snapshot_paths]
    print(f"Loaded {len(snapshot_paths)} snapshot(s) from {SNAPSHOTS_DIR}")

    kgs = _load_kgs(runner_mod, snapshot_paths)
    traces_complete = _traces_complete(runner_mod, labels)
    kgs = await _maybe_run_c_unseen(runner_mod, kgs, labels, traces_complete)

    detections = _extract_detections(runner_mod, kgs, labels)
    save(detections, OUTPUT_JSON)
    print(f"Wrote: {OUTPUT_JSON}")
    _print_summary(detections)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    print("=" * 50)
    print("  C-UNSEEN WEAK-SIGNAL DETECTOR")
    print("=" * 50)
    main()
