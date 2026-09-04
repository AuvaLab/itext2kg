#!/usr/bin/env python3
"""
Merge yearly OpenAI weak-signal DTKG snapshots and load the result into Neo4j.

Loads all pickles from datasets/c-unseen/snapshots/openai_weak_signal/, merges them
with ATOM ``parallel_atomic_merge``, optionally saves the merged KG, then pushes nodes
and relationships via ``Neo4jStorage.visualize_graph``.

Usage:
    pyenv activate venv-itext2kg-update
    python evaluation/merge_openai_weak_signal_snapshots_to_neo4j.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / ".git").exists():
        project_root = _p
        break
else:
    raise RuntimeError("Could not locate repository root")
sys.path.insert(0, str(project_root))

from itext2kg.atom.atom import Atom
from itext2kg.atom.models import KnowledgeGraph
from itext2kg.graph_integration import Neo4jStorage
from evaluation.c_unseen.paths import SNAPSHOTS

# ==========================
# User-configurable globals
# ==========================
SNAPSHOTS_DIR: Path = SNAPSHOTS
OUTPUT_MERGED_KG_PATH: Path = SNAPSHOTS_DIR / "merged_openai_weak_signal.pkl"

ENT_THRESHOLD: float = 0.8
REL_THRESHOLD: float = 0.7
MAX_WORKERS: int = -1  # -1 = all CPUs

NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME: str = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE: str | None = os.environ.get("NEO4J_DATABASE") or None

SAVE_MERGED_PICKLE: bool = True
CLEAR_NEO4J_BEFORE_LOAD: bool = False
PUSH_TO_NEO4J: bool = True


def resolve_max_workers(max_workers: int) -> int:
    if max_workers == -1:
        return max(1, os.cpu_count() or 1)
    return max(1, max_workers)


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


def load_snapshot_kg(path: Path) -> KnowledgeGraph | None:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "knowledge_graph" in obj:
        kg = obj["knowledge_graph"]
        return kg if isinstance(kg, KnowledgeGraph) else None

    if isinstance(obj, KnowledgeGraph):
        return obj

    return None


def load_yearly_snapshots() -> list[KnowledgeGraph]:
    if not SNAPSHOTS_DIR.exists():
        raise FileNotFoundError(f"Snapshots directory not found: {SNAPSHOTS_DIR}")

    snapshot_files = sorted(SNAPSHOTS_DIR.glob("*.pkl"))
    kgs: list[KnowledgeGraph] = []
    skipped = 0

    for path in snapshot_files:
        if path.name == OUTPUT_MERGED_KG_PATH.name:
            continue
        kg = load_snapshot_kg(path)
        if kg is None:
            skipped += 1
            continue
        if kg.is_empty():
            continue
        kgs.append(kg)

    if not kgs:
        raise FileNotFoundError(f"No yearly snapshot KGs found in {SNAPSHOTS_DIR}")

    print(f"Loaded {len(kgs)} snapshots ({skipped} skipped)")
    return kgs


def merge_snapshots(kgs: list[KnowledgeGraph]) -> KnowledgeGraph:
    atom = Atom(llm_model=None, embeddings_model=None)
    merged = atom.parallel_atomic_merge(
        kgs=kgs,
        existing_kg=None,
        ent_threshold=ENT_THRESHOLD,
        rel_threshold=REL_THRESHOLD,
        max_workers=resolve_max_workers(MAX_WORKERS),
    )
    print(
        f"Merged KG: {len(merged.entities)} entities, "
        f"{len(merged.relationships)} relationships"
    )
    return merged


def save_merged_kg(kg: KnowledgeGraph) -> None:
    OUTPUT_MERGED_KG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MERGED_KG_PATH, "wb") as f:
        pickle.dump(kg, f)
    print(f"Saved merged KG: {OUTPUT_MERGED_KG_PATH}")


def push_to_neo4j(kg: KnowledgeGraph) -> None:
    storage = Neo4jStorage(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=resolve_neo4j_password(),
        database=NEO4J_DATABASE,
    )
    if CLEAR_NEO4J_BEFORE_LOAD:
        print("Clearing Neo4j database...")
        storage.run_query("MATCH (n) DETACH DELETE n")

    print("Pushing merged KG to Neo4j...")
    storage.visualize_graph(kg)
    print("Neo4j load complete.")


def main() -> None:
    kgs = load_yearly_snapshots()
    merged_kg = merge_snapshots(kgs)

    if SAVE_MERGED_PICKLE:
        save_merged_kg(merged_kg)

    if PUSH_TO_NEO4J:
        push_to_neo4j(merged_kg)
    else:
        print("PUSH_TO_NEO4J=False; skipped Neo4j upload.")


if __name__ == "__main__":
    main()
