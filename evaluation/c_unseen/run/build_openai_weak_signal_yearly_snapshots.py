#!/usr/bin/env python3
"""
Build yearly DTKG snapshots from the OpenAI weak-signal benchmark.

Loads datasets/c-unseen/openai_weak_signal_benchmark.json, runs ATOM ``build_graph``
per year (``atomic_facts`` = fact texts, ``obs_timestamp`` = ``01-01-{year}``), and
writes one pickle per year under datasets/c-unseen/snapshots/openai_weak_signal/.

Usage:
    export OPENAI_API_KEY=...
    pyenv activate venv-itext2kg-update
    python evaluation/build_openai_weak_signal_yearly_snapshots.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / ".git").exists():
        project_root = _p
        break
else:
    raise RuntimeError("Could not locate repository root")
sys.path.insert(0, str(project_root))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from itext2kg.atom.atom import Atom
from evaluation.c_unseen.paths import BENCHMARK, SNAPSHOTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ==========================
# User-configurable globals
# ==========================
INPUT_JSON: Path = BENCHMARK
OUTPUT_DIR: Path = SNAPSHOTS
CHECKPOINT_FILE: Path = OUTPUT_DIR / "checkpoint.json"

# None = all years in the benchmark; otherwise restrict to this list of year strings.
YEARS: list[str] | None = None

ENT_THRESHOLD: float = 0.8
REL_THRESHOLD: float = 0.7
ENTITY_NAME_WEIGHT: float = 0.8
ENTITY_LABEL_WEIGHT: float = 0.2
MAX_WORKERS: int = 8

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set (required for OpenAI chat and embeddings).")

openai_llm_model = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4.1-2025-04-14",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

openai_embeddings_model = OpenAIEmbeddings(
    api_key=openai_api_key,
    model="text-embedding-3-large",
)


def load_checkpoint() -> set[str]:
    if not CHECKPOINT_FILE.exists():
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    done = data.get("completed_years", [])
    logger.info("Loaded checkpoint: %d years already done", len(done))
    return set(done)


def save_checkpoint(completed: set[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed_years": sorted(completed)}, f, indent=2)


def load_benchmark() -> dict:
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_JSON}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def year_obs_timestamp(year: str) -> str:
    return f"01-01-{year}"


def prepare_year_jobs(bench: dict, completed: set[str]) -> list[tuple[str, list[str], list[str]]]:
    year_to_facts: dict = bench.get("facts", {})
    all_years = sorted(year_to_facts.keys())
    if YEARS is not None:
        all_years = [y for y in all_years if y in YEARS]

    jobs: list[tuple[str, list[str], list[str]]] = []
    for year in all_years:
        if year in completed:
            continue
        facts = year_to_facts[year]
        texts = [str(f.get("text", "")).strip() for f in facts]
        domains = [str(f.get("domain", "") or "") for f in facts]
        if not any(texts):
            logger.info("Skipping %s: no fact texts", year)
            completed.add(year)
            continue
        jobs.append((year, texts, domains))
    return jobs


async def build_year_snapshot(
    atom: Atom,
    year: str,
    texts: list[str],
    domains: list[str],
) -> None:
    obs_ts = year_obs_timestamp(year)
    logger.info("Building snapshot for %s (%d facts)", year, len(texts))
    kg = await atom.build_graph(
        atomic_facts=texts,
        obs_timestamp=obs_ts,
        domains=domains,
        existing_knowledge_graph=None,
        ent_threshold=ENT_THRESHOLD,
        rel_threshold=REL_THRESHOLD,
        entity_name_weight=ENTITY_NAME_WEIGHT,
        entity_label_weight=ENTITY_LABEL_WEIGHT,
        max_workers=MAX_WORKERS,
    )
    out_path = OUTPUT_DIR / f"{year}.pkl"
    payload = {"year": year, "obs_timestamp": obs_ts, "knowledge_graph": kg}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(
        "Saved %s (%d entities, %d relationships)",
        out_path.name,
        len(kg.entities),
        len(kg.relationships),
    )


async def main() -> None:
    start = time.time()
    print("Loading benchmark:", INPUT_JSON)
    bench = load_benchmark()
    completed = load_checkpoint()
    jobs = prepare_year_jobs(bench, completed)

    if not jobs:
        save_checkpoint(completed)
        logger.info("Nothing left to process (checkpoint up to date).")
        print("No pending years. Checkpoint:", CHECKPOINT_FILE)
        return

    logger.info("Prepared %d years to build", len(jobs))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atom = Atom(llm_model=openai_llm_model, embeddings_model=openai_embeddings_model)

    results = await asyncio.gather(
        *[
            build_year_snapshot(atom, year, texts, domains)
            for year, texts, domains in jobs
        ],
        return_exceptions=True,
    )

    for (year, *_), result in zip(jobs, results):
        if isinstance(result, Exception):
            logger.error("Failed year %s: %s", year, result)
            raise result
        completed.add(year)

    save_checkpoint(completed)
    elapsed = time.time() - start
    logger.info("Done in %.1f s; %d yearly snapshots recorded.", elapsed, len(completed))
    print(f"Finished in {elapsed:.1f}s. Snapshots under: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("=" * 50)
    print("  YEARLY SNAPSHOTS — OPENAI WEAK-SIGNAL BENCHMARK (ATOM)")
    print("=" * 50)
    asyncio.run(main())
