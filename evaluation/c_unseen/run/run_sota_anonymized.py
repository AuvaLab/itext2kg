#!/usr/bin/env python3
"""
SOTA detectors on the anonymized Wiki-OpenAI benchmark.

Yoon and BERTrend read anonymized fact texts. BEAM and C-Unseen keep the original
DTKG topology and anonymize entity/relation surface forms (the entity_map's
intended prompt-time substitution). The scorer uses anonymized anchors and
remapped signal IDs (SS_01..SS_05).

BERTrend and C-Unseen: RUN_TIMES replicates. Yoon and BEAM: n=1.

Usage:
    pyenv activate venv-itext2kg-update
    python evaluation/c_unseen/run/run_sota_anonymized.py
    python evaluation/c_unseen/run/run_sota_anonymized.py --test-kg
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings

for _p in Path(__file__).resolve().parents:
    if (_p / ".git").exists():
        project_root = _p
        break
else:
    raise RuntimeError("Could not locate repository root")
sys.path.insert(0, str(project_root))

from evaluation.c_unseen.paths import (  # noqa: E402
    BENCHMARK_ANON,
    DETECTORS,
    ENTITY_MAP as ENTITY_MAP_PATH,
    MATCHING_SPEC,
    OUTPUTS,
    REPORTS,
    SCORING,
    SNAPSHOTS,
)

sys.path.insert(0, str(SCORING))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anonymize import apply_rules, build_rules, load  # noqa: E402
from run_sota_replicates import (  # noqa: E402
    BERTREND_SEEDS,
    RUN_TIMES,
    SCALAR_METRIC_KEYS,
    aggregate,
    load_bertrend,
    load_module,
    load_scorer,
    print_tables,
    run_bertrend_once,
    save_json,
    score_path,
)

# ==========================
# User-configurable globals
# ==========================
ANON_BENCHMARK: Path = BENCHMARK_ANON
ENTITY_MAP: Path = ENTITY_MAP_PATH
ORIG_ANCHORS: Path = MATCHING_SPEC
OUT_DIR: Path = OUTPUTS / "sota_anonymized_v2"
C_UNSEEN_REPORTS: Path = (
    REPORTS / "ablation" / "c_unseen_anon_quintuple_runs_grounded"
)
SNAPSHOTS_DIR: Path = SNAPSHOTS
# When True, re-run BEAM and C-Unseen (detections + LLM traces). Yoon/BERTrend
# still reuse existing outputs unless those files are missing.
FORCE_RERUN: bool = False
CENTRAL_ENTITY_SURFACE: str = "OpenAI"
TEST_KG_SUBGRAPH: bool = False
TEST_KG_YEAR: str = "2020"

# Spec tokens that are fragments of a mapped name, not a surface the entity
# map can safely replace in running text (e.g. "lp" is not "OpenAI LP").
ANCHOR_TOKEN_OVERRIDES: dict[str, str] = {
    "lp": "OrgVreinCharlie",
    "guild": "OrgSkaethas",
    "mulligan": "PersonKreask",
    "pbc": "OrgVreinEcho",
}

# Probes for the subgraph test (must be rewritten by the map).
KG_PROBE_SURFACES: tuple[str, ...] = (
    "sam altman",
    "elon musk",
    "tesla",
    "openai",
    "microsoft",
)
KG_ALLOWED_PLACEHOLDERS: tuple[str, ...] = (
    "orgvrein",
    "personkaundel",
    "personstyph",
    "orgnaendel",
    "orgmaumirn",
)


def make_text_replacer(emap: dict):
    rules, _ = build_rules(emap)
    aliases = {
        str(src).lower(): str(dst)
        for src, dst in emap.get("lowercase_aliases", {}).items()
    }

    def replace(text: str) -> str:
        if not text:
            return text
        out = apply_rules(str(text), rules, Counter())
        for src, dst in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
            out = re.sub(rf"(?<![\w]){re.escape(src)}(?![\w])", dst, out, flags=re.I)
        return out

    return replace


def make_anchor_replacer(emap: dict):
    """Case-insensitive, longest-first surface replacement (matching-spec / KG)."""
    skip = {"_meta", "replacement_rules", "residual_leakage_notes", "signal_ids"}
    pairs: list[tuple[str, str]] = []
    for category, entries in emap.items():
        if category in skip or not isinstance(entries, dict):
            continue
        for surface, placeholder in entries.items():
            pairs.append((str(surface), str(placeholder)))
    for src, dst in emap.get("lowercase_aliases", {}).items():
        pairs.append((str(src), str(dst)))
    pairs.sort(key=lambda kv: len(kv[0]), reverse=True)

    compiled = []
    for surface, placeholder in pairs:
        esc = re.escape(surface)
        left = r"(?<![\w])" if surface[0].isalnum() else r""
        right = r"(?![\w])" if surface[-1].isalnum() else r""
        compiled.append((re.compile(left + esc + right, re.I), placeholder))

    def replace(text: str) -> str:
        out = str(text)
        for pattern, placeholder in compiled:
            out = pattern.sub(placeholder, out)
        return out

    return replace


def make_kg_surface_replacer(emap: dict):
    """Case-insensitive map apply; snake_case labels are spaced before matching."""
    base = make_anchor_replacer(emap)

    def replace_field(text: str, *, is_label: bool = False) -> str:
        if not text:
            return text
        raw = str(text)
        spaced = raw.replace("_", " ") if is_label else raw
        return base(spaced)

    return replace_field


def write_anonymized_matching_spec(emap: dict, path: Path) -> dict[str, tuple[int, int]]:
    with open(ORIG_ANCHORS, "r", encoding="utf-8") as f:
        spec = json.load(f)
    id_map = emap.get("signal_ids", {})
    replace = make_anchor_replacer(emap)
    anon: dict = {"_meta": spec.get("_meta", {})}
    anon["_meta"]["anonymized"] = True
    orig_windows = spec.get("_meta", {}).get("windows", {})
    windows = {}
    for key, value in spec.items():
        if key.startswith("_"):
            continue
        new_key = id_map.get(key, key)
        if isinstance(value, list):
            anon[new_key] = [
                ANCHOR_TOKEN_OVERRIDES.get(rewritten.strip().lower(), rewritten)
                for rewritten in (replace(str(item)) for item in value)
            ]
        if key in orig_windows:
            windows[new_key] = orig_windows[key]
    anon["_meta"]["windows"] = windows
    save_json(anon, path)

    scorer_mod = load_module("weak_signal_scorer_windows", SCORING / "scorer.py")
    expected: dict[str, tuple[int, int]] = {}
    for old_id, bounds in scorer_mod.EXPECTED_WINDOWS.items():
        expected[id_map.get(old_id, old_id)] = bounds
    return expected


def anonymize_kg_surfaces(kg, replace_field) -> None:
    """Rewrite name, label, relation name, and atomic facts in-place."""
    seen: set[int] = set()

    def rewrite_entity(entity) -> None:
        if id(entity) in seen:
            return
        seen.add(id(entity))
        entity.name = replace_field(entity.name, is_label=False)
        if getattr(entity, "label", None) is not None:
            entity.label = replace_field(entity.label, is_label=True)

    for entity in getattr(kg, "entities", []):
        rewrite_entity(entity)
    for rel in getattr(kg, "relationships", []):
        rel.name = replace_field(rel.name, is_label=False)
        rewrite_entity(rel.startEntity)
        rewrite_entity(rel.endEntity)
        facts = list(getattr(rel.properties, "atomic_facts", []) or [])
        rel.properties.atomic_facts = [
            replace_field(str(f), is_label=False) for f in facts
        ]


def _unwrap_pickled_snapshot(obj):
    if isinstance(obj, dict) and "knowledge_graph" in obj:
        return obj["knowledge_graph"]
    return obj


def _format_triple(rel) -> str:
    start = rel.startEntity
    end = rel.endEntity
    props = getattr(rel, "properties", None)
    t_start = ""
    t_end = ""
    if props is not None:
        from datetime import datetime, timezone

        def _bound(values, pick: str) -> str:
            dates = []
            for value in list(values or []):
                if value in (None, "", 0, 0.0):
                    continue
                if isinstance(value, (int, float)):
                    try:
                        dates.append(datetime.fromtimestamp(float(value), tz=timezone.utc))
                    except (OSError, OverflowError, ValueError, TypeError):
                        continue
                    continue
                text = str(value).strip()
                if not text:
                    continue
                try:
                    from dateutil import parser as date_parser

                    parsed = date_parser.parse(text)
                except Exception:
                    continue
                if parsed is not None:
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    dates.append(parsed)
            if not dates:
                return ""
            chosen = min(dates) if pick == "start" else max(dates)
            return chosen.strftime("%Y-%m-%d")

        t_start = _bound(getattr(props, "t_start", []) or [], "start")
        t_end = _bound(getattr(props, "t_end", []) or [], "end")
    return (
        f"({start.name}: {start.label}) --> {rel.name} ({t_start}, {t_end}) --> "
        f"({end.name}: {end.label})"
    )


def _surface_in_field(field: str, surface: str) -> bool:
    """True if probe surface appears as a whole token (spaces/_ as separators)."""
    spaced = str(field).lower().replace("_", " ")
    pat = re.compile(
        rf"(?<![\w]){re.escape(surface.lower())}(?![\w])",
        re.I,
    )
    return pat.search(spaced) is not None


def _entity_hits_probe(entity, probes: tuple[str, ...]) -> bool:
    name = str(getattr(entity, "name", "") or "")
    label = str(getattr(entity, "label", "") or "")
    for surface in probes:
        if _surface_in_field(name, surface) or _surface_in_field(label, surface):
            return True
    return False


def _collect_probe_rel_indices(kg, probes: tuple[str, ...]) -> list[int]:
    hits: list[int] = []
    for idx, rel in enumerate(getattr(kg, "relationships", [])):
        if _entity_hits_probe(rel.startEntity, probes) or _entity_hits_probe(
            rel.endEntity, probes
        ):
            hits.append(idx)
    return hits


def _iter_entities(kg):
    seen: set[int] = set()
    for entity in getattr(kg, "entities", []):
        if id(entity) not in seen:
            seen.add(id(entity))
            yield entity
    for rel in getattr(kg, "relationships", []):
        for entity in (rel.startEntity, rel.endEntity):
            if id(entity) not in seen:
                seen.add(id(entity))
                yield entity


def _assert_no_probe_leaks(kg, probes: tuple[str, ...], allowed: tuple[str, ...]) -> None:
    allowed_set = {a.lower() for a in allowed}
    leaks: list[str] = []
    for entity in _iter_entities(kg):
        for field_name, raw in (
            ("name", entity.name),
            ("label", getattr(entity, "label", "")),
        ):
            text = str(raw or "")
            spaced = text.replace("_", " ")
            low = spaced.lower().strip()
            for surface in probes:
                if not _surface_in_field(spaced, surface):
                    continue
                if low in allowed_set or re.sub(r"[^a-z0-9]+", "", low) in allowed_set:
                    continue
                leaks.append(f"{field_name}={text!r} (probe={surface!r})")
    if leaks:
        sample = "\n  ".join(leaks[:20])
        raise AssertionError(
            f"KG still contains probe surfaces after anonymization "
            f"({len(leaks)} hit(s)):\n  {sample}"
        )


def test_kg_subgraph(emap: dict) -> None:
    """Load one yearly snapshot, print probe subgraph before/after, assert clean."""
    replace_field = make_kg_surface_replacer(emap)
    path = SNAPSHOTS_DIR / f"{TEST_KG_YEAR}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing snapshot for TEST_KG_YEAR={TEST_KG_YEAR}: {path}")

    with open(path, "rb") as f:
        kg = _unwrap_pickled_snapshot(pickle.load(f))

    n_ent = len(kg.entities)
    n_rel = len(kg.relationships)
    probe_idxs = _collect_probe_rel_indices(kg, KG_PROBE_SURFACES)
    if not probe_idxs:
        raise RuntimeError(
            f"No probe entities ({', '.join(KG_PROBE_SURFACES)}) in {path.name}. "
            "Try another TEST_KG_YEAR (e.g. 2023)."
        )

    print(f"Loaded {path.name}: {n_ent} entities, {n_rel} relationships")
    print(f"Probe subgraph edges: {len(probe_idxs)}")
    sample_idxs: list[int] = []
    seen_probes: set[str] = set()
    for idx in probe_idxs:
        rel = kg.relationships[idx]
        hits = [
            s
            for s in KG_PROBE_SURFACES
            if _entity_hits_probe(rel.startEntity, (s,))
            or _entity_hits_probe(rel.endEntity, (s,))
        ]
        if any(s not in seen_probes for s in hits) or len(sample_idxs) < 4:
            sample_idxs.append(idx)
            seen_probes.update(hits)
        if len(sample_idxs) >= 12:
            break
    print("\n--- BEFORE (sample) ---")
    for idx in sample_idxs:
        print(f"  [{idx}] {_format_triple(kg.relationships[idx])}")

    anonymize_kg_surfaces(kg, replace_field)

    print("\n--- AFTER (same edges) ---")
    for idx in sample_idxs:
        print(f"  [{idx}] {_format_triple(kg.relationships[idx])}")

    if len(kg.entities) != n_ent or len(kg.relationships) != n_rel:
        raise AssertionError(
            f"Topology changed: entities {n_ent}->{len(kg.entities)}, "
            f"relationships {n_rel}->{len(kg.relationships)}"
        )
    _assert_no_probe_leaks(kg, KG_PROBE_SURFACES, KG_ALLOWED_PLACEHOLDERS)
    print(
        f"\nOK: topology unchanged; no probe leaks in names/labels "
        f"(year={TEST_KG_YEAR})."
    )


def run_yoon(out_path: Path) -> None:
    yoon = load_module(
        "keyword_based_yoon",
        DETECTORS / "keyword_based_yoon.py",
    )
    yoon.INPUT_JSON = ANON_BENCHMARK
    yoon.OUTPUT_JSON = out_path
    yoon.main()


async def run_c_unseen_anonymized(replace_field, run_id: int, out_path: Path) -> list[dict]:
    if out_path.exists() and not FORCE_RERUN:
        print(f"[c_unseen anon run={run_id}] reusing {out_path.name}")
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    cu = load_module(
        "c_unseen_runner_mod",
        DETECTORS / "c_unseen_runner.py",
    )
    run_script = cu._load_run_script()

    orig_convert = run_script._to_signal_kg

    def convert_anon(src_kg):
        kg = orig_convert(src_kg)
        anonymize_kg_surfaces(kg, replace_field)
        return kg

    run_script._to_signal_kg = convert_anon
    cu.CENTRAL_ENTITY = replace_field(CENTRAL_ENTITY_SURFACE)
    reports = C_UNSEEN_REPORTS / f"run{run_id}"
    reports.mkdir(parents=True, exist_ok=True)
    cu.REPORTS_DIR = reports
    cu._configure_run_script(run_script)
    run_script.CENTRAL_ENTITY = cu.CENTRAL_ENTITY

    snapshot_paths = run_script._get_snapshot_paths()
    labels = [run_script._snapshot_label_from_path(p) for p in snapshot_paths]
    print(
        f"[c_unseen anon run={run_id}] {len(snapshot_paths)} snapshots; "
        f"central={cu.CENTRAL_ENTITY!r}; traces -> {reports}"
    )
    kgs = cu._load_kgs(run_script, snapshot_paths)
    traces_complete = cu._traces_complete(run_script, labels)
    kgs = await cu._maybe_run_c_unseen(run_script, kgs, labels, traces_complete)
    detections = cu._extract_detections(run_script, kgs, labels)
    save_json(detections, out_path)
    print(f"[c_unseen anon run={run_id}] {len(detections)} detections -> {out_path.name}")
    return detections


def run_beam_anonymized(replace_field, out_path: Path) -> None:
    beam = load_module(
        "graphlet_beam",
        DETECTORS / "graphlet_beam.py",
    )
    snapshots = beam.load_snapshots(beam.SNAPSHOTS_DIR)
    for _, kg in snapshots:
        anonymize_kg_surfaces(kg, replace_field)
    year_graphs = beam.build_year_graphs(snapshots)
    detections, vector_rows, yearly_stats = beam.extract_detections(year_graphs)
    beam.save(detections, out_path)
    print("Wrote:", out_path)
    beam._print_summary(vector_rows, yearly_stats, detections)


def main() -> None:
    import asyncio

    from api_keys import openai_api_key

    want_test = TEST_KG_SUBGRAPH or "--test-kg" in sys.argv
    emap = load(str(ENTITY_MAP))
    if want_test:
        print("=" * 50)
        print(f"  KG SUBGRAPH TEST (year={TEST_KG_YEAR})")
        print("=" * 50)
        test_kg_subgraph(emap)
        return

    print("=" * 50)
    print("  SOTA ON ANONYMIZED BENCHMARK")
    print("=" * 50)

    os.environ["OPENAI_API_KEY"] = openai_api_key
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    replace_kg = make_kg_surface_replacer(emap)
    anchors_path = OUT_DIR / "matching_spec_anonymized.json"
    expected = write_anonymized_matching_spec(emap, anchors_path)
    print("Anonymized matching spec:", anchors_path)
    print("Central entity:", replace_kg(CENTRAL_ENTITY_SURFACE))

    scorer = load_scorer()
    scorer.BENCHMARK_JSON = ANON_BENCHMARK
    scorer.ANCHORS_JSON = anchors_path
    scorer.EXPECTED_WINDOWS = expected
    with open(ANON_BENCHMARK, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    signals_meta = scorer.compute_windows(benchmark)
    anchors = scorer.load_anchors(anchors_path)
    print("Anonymized anchors:")
    for sid, words in anchors.items():
        print(f"  {sid}: {sorted(words)}")
    overrides = {}

    runs_by_method: dict[str, list[dict]] = {}

    yoon_path = OUT_DIR / "detections_yoon.json"
    print("\n=== Yoon (anonymized facts) ===")
    if yoon_path.exists():
        print(f"reusing {yoon_path.name}")
    else:
        run_yoon(yoon_path)
    runs_by_method["yoon"] = [
        score_path(scorer, "yoon", yoon_path, signals_meta, anchors, overrides)
    ]

    beam_path = OUT_DIR / "detections_beam.json"
    print("\n=== BEAM (anonymized KG labels) ===")
    if beam_path.exists() and not FORCE_RERUN:
        print(f"reusing {beam_path.name}")
    else:
        run_beam_anonymized(replace_kg, beam_path)
    runs_by_method["beam"] = [
        score_path(scorer, "beam", beam_path, signals_meta, anchors, overrides)
    ]

    print(f"\n=== BERTrend x{RUN_TIMES} (anonymized facts) ===")
    bertrend_paths = [OUT_DIR / f"detections_bertrend_run{i}.json" for i in range(RUN_TIMES)]
    if all(p.exists() for p in bertrend_paths):
        bertrend_runs: list[dict] = []
        for run_id, path in enumerate(bertrend_paths):
            print(f"  BERTrend run={run_id} reusing {path.name}")
            scored = score_path(scorer, "bertrend", path, signals_meta, anchors, overrides)
            save_json(scored, OUT_DIR / f"results_bertrend_run{run_id}.json")
            bertrend_runs.append(scored)
            print(f"    f1@k2={scored['k=2']['f1']:.3f}  f1@k3={scored['k=3']['f1']:.3f}")
    else:
        bt = load_bertrend()
        bt.INPUT_JSON = ANON_BENCHMARK
        facts = bt.load_data(bt.INPUT_JSON)
        slices = bt.build_year_slices(facts)
        embeddings_model = OpenAIEmbeddings(api_key=openai_api_key, model=bt.EMBEDDING_MODEL)
        emb_cache = OUT_DIR / "bertrend_embeddings.npz"
        if emb_cache.exists():
            print(f"reusing cached embeddings {emb_cache.name}")
            loaded = np.load(emb_cache)
            embeddings_by_year = {int(k): loaded[k] for k in loaded.files}
        else:
            print("Embedding anonymized facts once with", bt.EMBEDDING_MODEL)
            embeddings_by_year = bt.embed_all_facts(slices, embeddings_model)
            np.savez(emb_cache, **{str(year): matrix for year, matrix in embeddings_by_year.items()})
            print(f"wrote {emb_cache.name}")

        bertrend_runs = []
        for run_id, seed in enumerate(BERTREND_SEEDS):
            path = bertrend_paths[run_id]
            if path.exists():
                print(f"  BERTrend run={run_id} reusing {path.name}")
            else:
                print(f"  BERTrend run={run_id} seed={seed}")
                detections = run_bertrend_once(bt, slices, embeddings_by_year, embeddings_model, seed)
                bt.save(detections, path)
            scored = score_path(scorer, "bertrend", path, signals_meta, anchors, overrides)
            save_json(scored, OUT_DIR / f"results_bertrend_run{run_id}.json")
            bertrend_runs.append(scored)
            print(f"    f1@k2={scored['k=2']['f1']:.3f}  f1@k3={scored['k=3']['f1']:.3f}")
    runs_by_method["bertrend"] = bertrend_runs

    print(f"\n=== C-Unseen x{RUN_TIMES} (anonymized KG labels) ===")
    cunseen_runs: list[dict] = []
    for run_id in range(RUN_TIMES):
        path = OUT_DIR / f"detections_c_unseen_run{run_id}.json"
        if path.exists() and not FORCE_RERUN:
            print(f"[c_unseen anon run={run_id}] reusing {path.name}")
            with open(path, "r", encoding="utf-8") as f:
                detections = json.load(f)
        else:
            detections = asyncio.run(run_c_unseen_anonymized(replace_kg, run_id, path))
        scored = score_path(scorer, "c_unseen", path, signals_meta, anchors, overrides)
        save_json(scored, OUT_DIR / f"results_c_unseen_run{run_id}.json")
        cunseen_runs.append(scored)
        print(
            f"    n={len(detections)}  f1@k2={scored['k=2']['f1']:.3f}  "
            f"f1@k3={scored['k=3']['f1']:.3f}"
        )
    runs_by_method["c_unseen"] = cunseen_runs

    summary = aggregate(runs_by_method, scorer.K_LEVELS)
    print_tables(summary, scorer.K_LEVELS)

    payload = {
        "condition": "anonymized",
        "benchmark": str(ANON_BENCHMARK),
        "run_times": RUN_TIMES,
        "bertrend_seeds": BERTREND_SEEDS,
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
    out = OUT_DIR / "results_sota_anonymized_summary.json"
    save_json(payload, out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
