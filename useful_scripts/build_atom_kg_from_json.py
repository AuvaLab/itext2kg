#!/usr/bin/env python3
"""
Build an ATOM DTKG from a JSON list of dated texts.

1. Extract atomic facts for rows missing the field; write them back into the same JSON.
2. Build one KG snapshot per row (observation time = date field).
3. Merge snapshots into a final KG saved as JSON + NPZ (embeddings sidecar).
4. Optionally push the merged KG to Neo4j.

Fully resumable: rows with atomic_facts are skipped on re-run; existing snapshot
JSON files are skipped. Re-invoke safely after a crash.

Usage:
    pyenv activate venv-itext2kg-update
    python useful_scripts/build_atom_kg_from_json.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from itext2kg.atom.atom import Atom
from itext2kg.atom.models import KnowledgeGraph
from itext2kg.atom.models.schemas import AtomicFact
from itext2kg.graph_integration import Neo4jStorage
from itext2kg.llm_output_parsing import LangchainOutputParser
from itext2kg.logging_config import setup_logging

# ==========================
# User-configurable globals
# ==========================
INPUT_JSON_PATH = PROJECT_ROOT / "datasets" / "openai_demo" / "openai_temporal.json"
DATE_FIELD = "reception_date"  # <- rename here
TEXT_FIELD = "text"  # <- rename here
ATOMIC_FACTS_FIELD = "atomic_facts"  # written back into INPUT_JSON_PATH

SNAPSHOTS_DIR = INPUT_JSON_PATH.parent / "atom_snapshots"  # one .json+.npz per row
OUTPUT_KG_JSON = INPUT_JSON_PATH.parent / "openai_kg.json"
OUTPUT_KG_NPZ = INPUT_JSON_PATH.parent / "openai_kg.npz"
LOG_FILE = INPUT_JSON_PATH.parent / "build_atom_kg_from_json.log"

# None = process all rows; set to N for smoke / partial runs.
MAX_ROWS: int | None = None

SEND_TO_NEO4J = False
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "oNews")

OPENAI_MODEL_NAME = "gpt-4.1-2025-04-14"
OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-large"

ENT_THRESHOLD = 0.8
REL_THRESHOLD = 0.7
ENTITY_NAME_WEIGHT = 0.8
ENTITY_LABEL_WEIGHT = 0.2
MAX_WORKERS = 8  # threads inside Atom.build_graph merge
KG_CONCURRENCY = 8  # coroutines in flight for Atom.build_graph rows

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def configure_logging() -> logging.Logger:
    """Dual logging: console + file (script logger and itext2kg)."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # itext2kg internal loggers (Atom, matcher, parser, ...)
    setup_logging(
        level="INFO",
        format_string=LOG_FORMAT,
        log_file=str(LOG_FILE),
        console_output=True,
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Neo4j driver logs full Cypher (incl. embeddings) on CartesianProduct INFO
    # notifications — silence that flood; keep WARNING+.
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


logger = configure_logging()


def log_phase(title: str) -> None:
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


def resolve_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        from api_keys import openai_api_key

        if openai_api_key:
            return openai_api_key
    except ImportError:
        pass
    raise RuntimeError(
        "OpenAI API key not set. Export OPENAI_API_KEY or add openai_api_key to api_keys.py."
    )


def resolve_neo4j_password() -> str:
    if NEO4J_PASSWORD:
        return NEO4J_PASSWORD
    try:
        from api_keys import neo4_password

        return neo4_password
    except ImportError:
        pass
    raise RuntimeError(
        "Neo4j password not set. Export NEO4J_PASSWORD or add neo4_password to api_keys.py."
    )


def atomic_write_json(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_records() -> list[dict]:
    if not INPUT_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_JSON_PATH}")
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{INPUT_JSON_PATH} must be a JSON array of objects")
    return records


def work_indices(n_records: int) -> list[int]:
    if MAX_ROWS is None:
        return list(range(n_records))
    return list(range(min(MAX_ROWS, n_records)))


def snapshot_json_path(idx: int) -> Path:
    return SNAPSHOTS_DIR / f"{idx:05d}.json"


def snapshot_npz_path(idx: int) -> Path:
    return SNAPSHOTS_DIR / f"{idx:05d}.npz"


def snapshot_meta_path(idx: int) -> Path:
    return SNAPSHOTS_DIR / f"{idx:05d}.meta.json"


def snapshot_complete(idx: int) -> bool:
    """JSON + NPZ must both exist; otherwise treat as incomplete for resume."""
    return snapshot_json_path(idx).exists() and snapshot_npz_path(idx).exists()


def save_snapshot(idx: int, obs_timestamp, kg: KnowledgeGraph) -> None:
    """Persist one row snapshot as JSON + NPZ, with a tiny meta sidecar.

    Temp NPZ names must end in ``.npz`` so NumPy does not append another suffix.
    """
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = snapshot_json_path(idx)
    npz_path = snapshot_npz_path(idx)
    meta_path = snapshot_meta_path(idx)
    tmp_json = SNAPSHOTS_DIR / f".{idx:05d}.tmp.json"
    # Must end with ".npz" or NumPy appends another ".npz" suffix.
    tmp_npz = SNAPSHOTS_DIR / f".{idx:05d}.tmp.npz"
    tmp_meta = SNAPSHOTS_DIR / f".{idx:05d}.tmp.meta.json"

    kg.to_json(tmp_json, embeddings_path=tmp_npz)
    tmp_meta.write_text(
        json.dumps({"index": idx, "obs_timestamp": obs_timestamp}, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_json, json_path)
    os.replace(tmp_npz, npz_path)
    os.replace(tmp_meta, meta_path)


async def extract_atomic_facts(
    parser: LangchainOutputParser, records: list[dict]
) -> None:
    log_phase("PHASE 1/4 — Atomic facts extraction")
    indices = work_indices(len(records))
    already_done = sum(1 for i in indices if ATOMIC_FACTS_FIELD in records[i])
    todo = [(i, records[i]) for i in indices if ATOMIC_FACTS_FIELD not in records[i]]

    logger.info(
        "Work rows=%d | already have %s=%d | to extract=%d",
        len(indices),
        ATOMIC_FACTS_FIELD,
        already_done,
        len(todo),
    )
    if not todo:
        logger.info("Nothing to extract — skipping to next phase.")
        return

    logger.info("Calling LLM in parallel batches for %d contexts...", len(todo))
    t0 = time.time()
    contexts = [str(r.get(TEXT_FIELD) or "") for _, r in todo]
    results = await parser.extract_information_as_json_for_context(
        AtomicFact, contexts, return_exceptions=True
    )

    n_ok = 0
    n_fail = 0
    n_facts = 0
    for (i, r), out in zip(todo, results):
        if isinstance(out, BaseException):
            logger.warning("row %d/%d atomic-facts FAILED: %s", i, len(records) - 1, out)
            n_fail += 1
            continue
        facts = out.atomic_fact if out else []
        r[ATOMIC_FACTS_FIELD] = facts
        n_ok += 1
        n_facts += len(facts)
        logger.info(
            "row %d/%d OK — %d atomic facts (elapsed %.1fs)",
            i,
            len(records) - 1,
            len(facts),
            time.time() - t0,
        )

    atomic_write_json(records, INPUT_JSON_PATH)
    logger.info(
        "Phase 1 done in %.1fs — wrote %d successes (%d total facts), %d failures left for resume → %s",
        time.time() - t0,
        n_ok,
        n_facts,
        n_fail,
        INPUT_JSON_PATH,
    )


async def build_snapshots(atom: Atom, records: list[dict]) -> None:
    log_phase("PHASE 2/4 — Per-row KG snapshots")
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(KG_CONCURRENCY)
    indices = work_indices(len(records))

    already = sum(1 for i in indices if snapshot_complete(i))
    missing_facts = sum(
        1
        for i in indices
        if not snapshot_complete(i) and ATOMIC_FACTS_FIELD not in records[i]
    )
    todo_count = sum(
        1
        for i in indices
        if not snapshot_complete(i) and ATOMIC_FACTS_FIELD in records[i]
    )
    logger.info(
        "Work rows=%d | snapshots already done=%d | missing atomic_facts=%d | to build=%d | concurrency=%d",
        len(indices),
        already,
        missing_facts,
        todo_count,
        KG_CONCURRENCY,
    )
    if todo_count == 0:
        logger.info("Nothing to build — skipping to next phase.")
        return

    done_counter = {"n": 0}
    t0 = time.time()

    async def build_one(idx: int) -> None:
        row = records[idx]
        if snapshot_complete(idx) or ATOMIC_FACTS_FIELD not in row:
            return
        facts = row[ATOMIC_FACTS_FIELD] or []
        logger.info(
            "Building snapshot %05d (%d facts, obs=%s) ...",
            idx,
            len(facts),
            row.get(DATE_FIELD),
        )
        async with sem:
            if not facts:
                kg = KnowledgeGraph()
                logger.warning("row %d: empty atomic_facts -> empty snapshot", idx)
            else:
                kg = await atom.build_graph(
                    atomic_facts=facts,
                    obs_timestamp=str(row[DATE_FIELD]),
                    ent_threshold=ENT_THRESHOLD,
                    rel_threshold=REL_THRESHOLD,
                    entity_name_weight=ENTITY_NAME_WEIGHT,
                    entity_label_weight=ENTITY_LABEL_WEIGHT,
                    max_workers=MAX_WORKERS,
                )
        save_snapshot(idx, row[DATE_FIELD], kg)
        done_counter["n"] += 1
        logger.info(
            "[%d/%d] Saved snapshot %s (+ .npz) (%d entities, %d relationships) — elapsed %.1fs",
            done_counter["n"],
            todo_count,
            snapshot_json_path(idx).name,
            len(kg.entities),
            len(kg.relationships),
            time.time() - t0,
        )

    results = await asyncio.gather(
        *[build_one(i) for i in indices],
        return_exceptions=True,
    )
    n_fail = 0
    for i, res in zip(indices, results):
        if isinstance(res, BaseException):
            logger.warning("row %d snapshot FAILED: %s", i, res)
            n_fail += 1
    logger.info(
        "Phase 2 done in %.1fs — built %d snapshots, %d failures left for resume → %s",
        time.time() - t0,
        done_counter["n"],
        n_fail,
        SNAPSHOTS_DIR,
    )


def load_snapshot_kgs(indices: list[int] | None = None) -> list[KnowledgeGraph]:
    if indices is None:
        paths = sorted(SNAPSHOTS_DIR.glob("*.json"))
        paths = [p for p in paths if not p.name.endswith(".meta.json")]
    else:
        paths = [snapshot_json_path(i) for i in indices if snapshot_complete(i)]

    kgs: list[KnowledgeGraph] = []
    for path in paths:
        if path.name.endswith(".tmp"):
            continue
        npz_path = path.with_suffix(".npz")
        kg = KnowledgeGraph.from_json(
            path,
            embeddings_path=npz_path if npz_path.exists() else None,
        )
        if isinstance(kg, KnowledgeGraph) and not kg.is_empty():
            kgs.append(kg)
    return kgs


def merge_and_save(atom: Atom, records: list[dict]) -> KnowledgeGraph:
    log_phase("PHASE 3/4 — Merge snapshots → final KG")
    indices = work_indices(len(records))
    kgs = load_snapshot_kgs(indices)
    t0 = time.time()
    if not kgs:
        logger.warning("No non-empty snapshots found; writing empty KG")
        final_kg = KnowledgeGraph()
    else:
        logger.info("Merging %d non-empty snapshots (workers=%d)...", len(kgs), MAX_WORKERS)
        final_kg = atom.parallel_atomic_merge(
            kgs=kgs,
            rel_threshold=REL_THRESHOLD,
            ent_threshold=ENT_THRESHOLD,
            max_workers=MAX_WORKERS,
        )
    OUTPUT_KG_JSON.parent.mkdir(parents=True, exist_ok=True)
    final_kg.to_json(OUTPUT_KG_JSON, embeddings_path=OUTPUT_KG_NPZ)
    logger.info(
        "Phase 3 done in %.1fs — saved %s (+ %s) (%d entities, %d relationships)",
        time.time() - t0,
        OUTPUT_KG_JSON,
        OUTPUT_KG_NPZ.name,
        len(final_kg.entities),
        len(final_kg.relationships),
    )
    return final_kg


def push_to_neo4j(kg: KnowledgeGraph) -> None:
    log_phase("PHASE 4/4 — Push to Neo4j")
    logger.info(
        "Connecting to Neo4j uri=%s user=%s database=%s",
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_DATABASE,
    )
    t0 = time.time()
    storage = Neo4jStorage(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=resolve_neo4j_password(),
        database=NEO4J_DATABASE,
    )
    logger.info(
        "Pushing KG (%d entities, %d relationships)...",
        len(kg.entities),
        len(kg.relationships),
    )
    storage.visualize_graph(kg)
    logger.info("Phase 4 done in %.1fs — Neo4j load complete.", time.time() - t0)


async def main() -> None:
    start = time.time()
    log_phase("BUILD ATOM KG FROM JSON — start")
    logger.info("Log file: %s", LOG_FILE.resolve())
    logger.info("Input JSON: %s", INPUT_JSON_PATH)
    logger.info("Snapshots dir: %s", SNAPSHOTS_DIR)
    logger.info("Output KG: %s (+ %s)", OUTPUT_KG_JSON, OUTPUT_KG_NPZ.name)
    logger.info(
        "Config: MAX_ROWS=%s | SEND_TO_NEO4J=%s | model=%s | KG_CONCURRENCY=%d",
        MAX_ROWS,
        SEND_TO_NEO4J,
        OPENAI_MODEL_NAME,
        KG_CONCURRENCY,
    )

    records = load_records()
    work = work_indices(len(records))
    logger.info("Loaded %d records (%d in work set)", len(records), len(work))

    logger.info("Bootstrapping OpenAI LLM + embeddings...")
    api_key = resolve_openai_api_key()
    openai_llm = ChatOpenAI(
        api_key=api_key,
        model=OPENAI_MODEL_NAME,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    openai_emb = OpenAIEmbeddings(api_key=api_key, model=OPENAI_EMBEDDINGS_MODEL)
    parser = LangchainOutputParser(llm_model=openai_llm, embeddings_model=openai_emb)
    atom = Atom(llm_model=openai_llm, embeddings_model=openai_emb)
    logger.info("Models ready.")

    await extract_atomic_facts(parser, records)
    # Reload from disk so in-memory records match what was persisted
    # (and so resume mid-file still sees written atomic_facts).
    records = load_records()

    await build_snapshots(atom, records)
    final_kg = merge_and_save(atom, records)

    if SEND_TO_NEO4J:
        push_to_neo4j(final_kg)
    else:
        log_phase("PHASE 4/4 — Neo4j skipped")
        logger.info("SEND_TO_NEO4J=False; skipped Neo4j upload.")

    elapsed = time.time() - start
    log_phase("DONE")
    logger.info(
        "Finished in %.1fs | merged KG: %s (+ %s) | log: %s",
        elapsed,
        OUTPUT_KG_JSON,
        OUTPUT_KG_NPZ.name,
        LOG_FILE,
    )


if __name__ == "__main__":
    asyncio.run(main())
