#!/usr/bin/env python3
"""
Run C-Unseen in whole_snapshot mode on a sequence of KG snapshots.

For each snapshot:
1. Sends the full snapshot (numbered triples) to the LLM for rare/precursor detection.
2. Compares minimum bridging subgraphs across snapshots for weak-signal detection.
3. Writes per-snapshot LLM traces under REPORTS_DIR.

Usage:
    pyenv activate venv-itext2kg-update
    python evaluation/c_unseen/run/run_c_unseen_whole_snapshot.py
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import re
import sys
import textwrap
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

for _p in Path(__file__).resolve().parents:
    if (_p / ".git").exists():
        PROJECT_ROOT = _p
        break
else:
    raise RuntimeError("Could not locate repository root")
sys.path.insert(0, str(PROJECT_ROOT))

from api_keys import openai_api_key  # noqa: E402
from evaluation.c_unseen.paths import REPORTS, SNAPSHOTS  # noqa: E402
from itext2kg.c_unseen.models import (  # noqa: E402
    SignalKnowledgeGraph,
    SignalProperties,
    SignalRelationship,
)
from itext2kg.c_unseen.rare_elements_detector.detector import (  # noqa: E402
    format_quintuple,
)

# ============================================================================
# GLOBAL CONFIGURATION (edit these)
# ============================================================================

# Directory of snapshot pickles (e.g. datasets/c-unseen/snapshots/openai_weak_signal/2015.pkl)
SNAPSHOTS_DIR = SNAPSHOTS
SNAPSHOT_GLOB = "*.pkl"

# Inclusive year range when filenames are {year}.pkl; set both to None to load all matches.
YEAR_START: str | None = "2015"
YEAR_END: str | None = "2025"

# Entity name passed to the rare-detection prompt for context.
CENTRAL_ENTITY = "openai"

# Per-snapshot LLM traces: whole_snapshot_rare_{label}.txt, weak_signal_bridging_{label}.txt
REPORTS_DIR = REPORTS

# High-level run summary.
REPORT_OUTPUT_PATH = REPORTS_DIR / "c_unseen_whole_snapshot_summary.txt"

# Optionally persist processed KGs (with rare / weak_signal_pred flags) after the run.
SAVE_PROCESSED_SNAPSHOTS = False
PROCESSED_OUTPUT_DIR = SNAPSHOTS_DIR / "c_unseen_processed"

# OpenAI models
OPENAI_MODEL_NAME = "gpt-5.4-mini-2026-03-17"
OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-large"
TEMPERATURE = 0
MAX_TOKENS = None
TIMEOUT = None
MAX_RETRIES = 2

_YEAR_STEM_RE = re.compile(r"^(\d{4})$")
# OpenAI reasoning models (o1/o3/o4-*) reject the temperature parameter.
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")
# These models also reject the parallel_tool_calls parameter LangChain sends by
# default for structured output.
_PARALLEL_TOOL_CALLS_UNSUPPORTED_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _model_supports_temperature(model_name: str) -> bool:
    name = model_name.lower().strip()
    return not any(name.startswith(prefix) for prefix in _REASONING_MODEL_PREFIXES)


def _model_disables_parallel_tool_calls(model_name: str) -> bool:
    name = model_name.lower().strip()
    return any(
        name.startswith(prefix)
        for prefix in _PARALLEL_TOOL_CALLS_UNSUPPORTED_PREFIXES
    )


def _build_chat_openai(**kwargs) -> ChatOpenAI:
    model = str(kwargs.get("model", ""))
    if not _model_supports_temperature(model):
        kwargs.pop("temperature", None)
    if _model_disables_parallel_tool_calls(model):
        disabled = dict(kwargs.pop("disabled_params", None) or {})
        disabled["parallel_tool_calls"] = None
        kwargs["disabled_params"] = disabled
    return ChatOpenAI(**kwargs)


def _unwrap_pickled_snapshot(obj):
    if isinstance(obj, dict) and "knowledge_graph" in obj:
        return obj["knowledge_graph"]
    return obj


def _to_signal_kg(src_kg) -> SignalKnowledgeGraph:
    """Re-wrap an ATOM KnowledgeGraph as a SignalKnowledgeGraph with cleared flags."""
    from itext2kg.atom.models import Entity

    entity_map: dict[tuple[str, str], Entity] = {}
    entities: list[Entity] = []

    for entity in src_kg.entities:
        key = (entity.name, entity.label)
        if key in entity_map:
            continue
        entity_map[key] = entity
        entities.append(entity)

    relationships: list[SignalRelationship] = []
    for rel in src_kg.relationships:
        props = rel.properties
        relationships.append(
            SignalRelationship(
                name=rel.name,
                startEntity=entity_map[(rel.startEntity.name, rel.startEntity.label)],
                endEntity=entity_map[(rel.endEntity.name, rel.endEntity.label)],
                properties=SignalProperties(
                    embeddings=getattr(props, "embeddings", None),
                    atomic_facts=list(getattr(props, "atomic_facts", []) or []),
                    domains=list(getattr(props, "domains", []) or []),
                    t_obs=list(getattr(props, "t_obs", []) or []),
                    t_start=list(getattr(props, "t_start", []) or []),
                    t_end=list(getattr(props, "t_end", []) or []),
                    rare=False,
                    weak_signal_pred=False,
                    already_corroborated=False,
                ),
            )
        )

    return SignalKnowledgeGraph(entities=entities, relationships=relationships)


def _snapshot_label_from_path(path: Path) -> str:
    return path.stem


def _get_snapshot_paths() -> list[Path]:
    files = sorted(SNAPSHOTS_DIR.glob(SNAPSHOT_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found in {SNAPSHOTS_DIR} with {SNAPSHOT_GLOB}")

    if YEAR_START is None and YEAR_END is None:
        return files

    start = YEAR_START or YEAR_END
    end = YEAR_END or YEAR_START
    if start is None or end is None:
        return files
    if end < start:
        raise ValueError("YEAR_END must be >= YEAR_START")

    selected: list[Path] = []
    for path in files:
        m = _YEAR_STEM_RE.match(path.stem)
        if not m:
            continue
        year = m.group(1)
        if start <= year <= end:
            selected.append(path)
    if not selected:
        raise IndexError(
            f"No snapshot files in year range [{start}, {end}] under {SNAPSHOTS_DIR}"
        )
    return selected


def _format_triple(rel) -> str:
    return format_quintuple(rel)


def _extract_response_json(trace_path: Path) -> dict | None:
    if not trace_path.exists():
        return None
    content = trace_path.read_text(encoding="utf-8")
    if "=== RESPONSE ===" not in content:
        return None
    block = content.split("=== RESPONSE ===", 1)[1]
    for stop in ("=== FLAGGED ===", "=== NOTE ==="):
        if stop in block:
            block = block.split(stop, 1)[0]
    block = block.strip()
    if not block or block == "(no response)":
        return None
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return None


def _wrap_paragraph(text: str, indent: str = "      ", width: int = 96) -> list[str]:
    text = (text or "").strip()
    if not text:
        return [f"{indent}(empty)"]
    return textwrap.wrap(
        text,
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def _normalize_index_list(raw) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [raw]
    return [int(i) for i in raw]


def _rare_buckets_from_trace(kg, label: str) -> tuple[str, list[dict]]:
    data = _extract_response_json(REPORTS_DIR / f"whole_snapshot_rare_{label}.txt")
    baseline = (data or {}).get("baseline_pattern", "")
    buckets: list[dict] = []

    if data and data.get("explanations"):
        for bucket_id, entry in enumerate(data["explanations"], 1):
            indices = _normalize_index_list(entry.get("index"))
            triples = [
                (idx, _format_triple(kg.relationships[idx]))
                for idx in indices
                if 0 <= idx < len(kg.relationships)
            ]
            buckets.append(
                {
                    "bucket_id": bucket_id,
                    "indices": indices,
                    "why_rare": entry.get("why_rare", ""),
                    "monitor_priority": entry.get("monitor_priority", "low"),
                    "triples": triples,
                }
            )
    else:
        rare_indices = (
            _normalize_index_list((data or {}).get("rare_indices"))
            if data
            else [i for i, r in enumerate(kg.relationships) if r.properties.rare]
        )
        for bucket_id, idx in enumerate(rare_indices, 1):
            if 0 <= idx < len(kg.relationships):
                buckets.append(
                    {
                        "bucket_id": bucket_id,
                        "indices": [idx],
                        "why_rare": "(no explanation in trace)",
                        "monitor_priority": "n/a",
                        "triples": [(idx, _format_triple(kg.relationships[idx]))],
                    }
                )

    return baseline, buckets


def _weak_buckets_from_trace(kg, label: str) -> list[dict]:
    data = _extract_response_json(REPORTS_DIR / f"weak_signal_bridging_{label}.txt")
    if not data:
        weak_indices = [i for i, r in enumerate(kg.relationships) if r.properties.weak_signal_pred]
        return [
            {
                "bucket_id": i,
                "bridging_indices": [],
                "kg_indices": [idx],
                "theme": "(no explanation in trace)",
                "explanation": "",
                "triples": [(None, idx, _format_triple(kg.relationships[idx]))],
            }
            for i, idx in enumerate(weak_indices, 1)
            if 0 <= idx < len(kg.relationships)
        ]

    current_rare = [i for i, r in enumerate(kg.relationships) if r.properties.rare]
    current_bridging = kg.extract_connecting_subgraph(current_rare)
    buckets: list[dict] = []

    if data.get("explanations"):
        for bucket_id, entry in enumerate(data["explanations"], 1):
            bridging_indices = _normalize_index_list(entry.get("index"))
            triples: list[tuple[int | None, int, str]] = []
            kg_indices: list[int] = []
            for j in bridging_indices:
                if 0 <= j < len(current_bridging):
                    kg_idx = current_bridging[j]
                    kg_indices.append(kg_idx)
                    triples.append((j, kg_idx, _format_triple(kg.relationships[kg_idx])))
            buckets.append(
                {
                    "bucket_id": bucket_id,
                    "bridging_indices": bridging_indices,
                    "kg_indices": kg_indices,
                    "theme": entry.get("theme", ""),
                    "explanation": entry.get("explanation", ""),
                    "triples": triples,
                }
            )
    else:
        weak_bridging = _normalize_index_list(data.get("weak_signal_indices"))
        for bucket_id, j in enumerate(weak_bridging, 1):
            if 0 <= j < len(current_bridging):
                kg_idx = current_bridging[j]
                buckets.append(
                    {
                        "bucket_id": bucket_id,
                        "bridging_indices": [j],
                        "kg_indices": [kg_idx],
                        "theme": "(no explanation in trace)",
                        "explanation": "",
                        "triples": [(j, kg_idx, _format_triple(kg.relationships[kg_idx]))],
                    }
                )

    return buckets


def _snapshot_summary(label: str, kg) -> dict:
    baseline, rare_buckets = _rare_buckets_from_trace(kg, label)
    weak_buckets = _weak_buckets_from_trace(kg, label)
    rare_triple_count = sum(len(b["indices"]) for b in rare_buckets)
    weak_triple_count = sum(len(b["kg_indices"]) for b in weak_buckets)
    return {
        "label": label,
        "relationships": len(kg.relationships),
        "baseline_pattern": baseline,
        "rare_bucket_count": len(rare_buckets),
        "rare_triple_count": rare_triple_count,
        "weak_bucket_count": len(weak_buckets),
        "weak_triple_count": weak_triple_count,
        "rare_buckets": rare_buckets,
        "weak_buckets": weak_buckets,
    }


def _section(title: str, width: int = 80) -> list[str]:
    return ["", title, "-" * len(title)]


def _build_summary_report(summaries: list[dict]) -> str:
    lines = [
        "C-UNSEEN WHOLE-SNAPSHOT RUN SUMMARY",
        "=" * 80,
        f"Central entity : {CENTRAL_ENTITY}",
        f"Snapshots dir  : {SNAPSHOTS_DIR}",
        f"Reports dir    : {REPORTS_DIR}",
        f"Year range     : {YEAR_START or 'all'} .. {YEAR_END or 'all'}",
        "",
    ]

    for s in summaries:
        lines.extend(
            [
                "",
                "=" * 80,
                f"SNAPSHOT {s['label']}",
                "=" * 80,
                f"Relationships  : {s['relationships']}",
                f"Rare elements  : {s['rare_bucket_count']} bucket(s), {s['rare_triple_count']} triple(s)",
                f"Weak signals   : {s['weak_bucket_count']} bucket(s), {s['weak_triple_count']} triple(s)",
            ]
        )

        lines.extend(_section("Baseline narrative"))
        if s["baseline_pattern"]:
            lines.extend(_wrap_paragraph(s["baseline_pattern"], indent="  "))
        else:
            lines.append("  (not available)")

        lines.extend(_section(f"Rare elements ({s['rare_bucket_count']})"))
        if not s["rare_buckets"]:
            lines.append("  (none)")
        else:
            for bucket in s["rare_buckets"]:
                idx_label = ", ".join(str(i) for i in bucket["indices"])
                lines.append("")
                lines.append(
                    f"  [Rare bucket {bucket['bucket_id']}]  "
                    f"priority={bucket['monitor_priority']}  indices=[{idx_label}]"
                )
                lines.append("  Why rare:")
                lines.extend(_wrap_paragraph(bucket["why_rare"], indent="    "))
                lines.append("  Triples:")
                for idx, triple in bucket["triples"]:
                    lines.append(f"    - [{idx}] {triple}")

        lines.extend(_section(f"Weak signals ({s['weak_bucket_count']})"))
        if not s["weak_buckets"]:
            lines.append("  (none)")
        else:
            for bucket in s["weak_buckets"]:
                bridging_label = ", ".join(str(i) for i in bucket["bridging_indices"])
                kg_label = ", ".join(str(i) for i in bucket["kg_indices"])
                lines.append("")
                lines.append(
                    f"  [Weak-signal bucket {bucket['bucket_id']}]  "
                    f"bridging=[{bridging_label}]  kg_indices=[{kg_label}]"
                )
                lines.append("  Theme:")
                lines.extend(_wrap_paragraph(bucket["theme"], indent="    "))
                lines.append("  Explanation:")
                lines.extend(_wrap_paragraph(bucket["explanation"], indent="    "))
                lines.append("  Triples:")
                for bridging_idx, kg_idx, triple in bucket["triples"]:
                    if bridging_idx is None:
                        lines.append(f"    - [kg {kg_idx}] {triple}")
                    else:
                        lines.append(f"    - [bridging {bridging_idx} -> kg {kg_idx}] {triple}")

    lines.extend(
        [
            "",
            "=" * 80,
            "Per-snapshot LLM traces",
            "=" * 80,
        ]
    )
    for s in summaries:
        label = s["label"]
        lines.append(f"  {label}:")
        lines.append(f"    - {REPORTS_DIR / f'whole_snapshot_rare_{label}.txt'}")
        lines.append(f"    - {REPORTS_DIR / f'weak_signal_bridging_{label}.txt'}")

    return "\n".join(lines).rstrip() + "\n"


async def main() -> None:
    if not openai_api_key:
        raise RuntimeError("openai_api_key is empty in api_keys.py")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    from itext2kg.c_unseen import CUnseen

    snapshot_paths = _get_snapshot_paths()
    snapshot_labels = [_snapshot_label_from_path(p) for p in snapshot_paths]

    print(f"Running C-Unseen on {len(snapshot_paths)} snapshot(s)")
    print(f"Central entity: {CENTRAL_ENTITY}")
    print(f"Reports dir: {REPORTS_DIR}")
    for path, label in zip(snapshot_paths, snapshot_labels):
        print(f"  - {label}: {path.name}")

    llm_kwargs = {
        "api_key": openai_api_key,
        "model": OPENAI_MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "timeout": TIMEOUT,
        "max_retries": MAX_RETRIES,
    }
    if _model_supports_temperature(OPENAI_MODEL_NAME):
        llm_kwargs["temperature"] = TEMPERATURE
    else:
        print(f"Model {OPENAI_MODEL_NAME!r} does not support temperature — omitting it.")
    llm_model = _build_chat_openai(**llm_kwargs)
    embeddings_model = OpenAIEmbeddings(
        api_key=openai_api_key,
        model=OPENAI_EMBEDDINGS_MODEL,
    )

    runner = CUnseen(
        llm_model=llm_model,
        embeddings_model=embeddings_model,
        reports_dir=REPORTS_DIR,
    )

    kgs = []
    for path in snapshot_paths:
        with open(path, "rb") as f:
            src_kg = _unwrap_pickled_snapshot(pickle.load(f))
        if not hasattr(src_kg, "entities") or not hasattr(src_kg, "relationships"):
            raise TypeError(
                f"{path}: expected KnowledgeGraph-like object, got {type(src_kg).__name__}"
            )
        kgs.append(_to_signal_kg(src_kg))

    processed = await runner.process_snapshots(
        kgs,
        snapshot_labels=snapshot_labels,
        central_entity_name=CENTRAL_ENTITY,
    )

    summaries = []
    for kg, label in zip(processed, snapshot_labels):
        summary = _snapshot_summary(label, kg)
        summaries.append(summary)
        print(
            f"\nSnapshot {label}: "
            f"relationships={summary['relationships']} | "
            f"rare={summary['rare_bucket_count']} bucket(s)/{summary['rare_triple_count']} triple(s) | "
            f"weak_signal={summary['weak_bucket_count']} bucket(s)/{summary['weak_triple_count']} triple(s)"
        )
        for bucket in summary["weak_buckets"]:
            kg_label = ", ".join(str(i) for i in bucket["kg_indices"])
            theme = bucket["theme"] or "(empty)"
            preview = theme if len(theme) <= 80 else f"{theme[:77]}..."
            print(f"  weak bucket {bucket['bucket_id']} [kg {kg_label}] theme={preview}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_text = _build_summary_report(summaries)
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nSummary report: {REPORT_OUTPUT_PATH}")

    if SAVE_PROCESSED_SNAPSHOTS:
        PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for kg, label in zip(processed, snapshot_labels):
            out_path = PROCESSED_OUTPUT_DIR / f"{label}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(kg, f)
            print(f"Saved processed snapshot: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
