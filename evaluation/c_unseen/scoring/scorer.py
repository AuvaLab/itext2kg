#!/usr/bin/env python3
"""
Shared scorer for Wiki-OpenAI weak-signal benchmark detections.

Computes detection windows from the benchmark, matches detections to strong signals
via shared anchor-word counts at k=1,2,3, and grades Yoon / BERTrend / BEAM / C-Unseen outputs
identically. Python standard library only.

Usage:
    pyenv activate venv-itext2kg-update
    python evaluation/c_unseen/scoring/scorer.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / ".git").exists():
        project_root = _p
        break
else:
    raise RuntimeError("Could not locate repository root")
sys.path.insert(0, str(project_root))

from evaluation.c_unseen.paths import (  # noqa: E402
    BENCHMARK,
    MATCHING_SPEC,
    OUTPUTS,
    OVERRIDES,
)

# ==========================
# User-configurable globals
# ==========================
BENCHMARK_JSON: Path = BENCHMARK
ANCHORS_JSON: Path = MATCHING_SPEC
OVERRIDES_JSON: Path = OVERRIDES
OUTPUT_DIR: Path = OUTPUTS

DETECTION_FILES: list[tuple[str, Path]] = [
    ("yoon", OUTPUT_DIR / "detections_yoon.json"),
    ("bertrend", OUTPUT_DIR / "detections_bertrend.json"),
    ("beam", OUTPUT_DIR / "detections_beam.json"),
    ("c_unseen", OUTPUT_DIR / "detections_c_unseen.json"),
]

LEAD_DATE_CONVENTION: str = "year_end"
N_EVENTS: int = 5
K_LEVELS: list[int] = [1, 2, 3]
DESC_FIELDS: list[str] = ["description", "keyword", "label"]

EXPECTED_WINDOWS: dict[str, tuple[int, int]] = {
    "SS_FORPROFIT_2019": (2015, 2019),
    "SS_BOARD_COUP_2023": (2019, 2023),
    "SS_NYT_LAWSUIT_2023": (2021, 2023),
    "SS_DEFENCE_TURN": (2019, 2025),
    "SS_FORPROFIT_PBC_2025": (2017, 2025),
}

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def parse_partial_date(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None

    if len(text) == 4 and text.isdigit():
        return date(int(text), 1, 1)

    if len(text) == 7 and text[4] == "-":
        year, month = text.split("-")
        return date(int(year), int(month), 1)

    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        pass

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.date()
    except ValueError as exc:
        raise ValueError(f"Unparseable date: {value!r}") from exc


def compute_windows(benchmark: dict) -> dict[str, dict]:
    signals_meta: dict[str, dict] = {}

    for signal in benchmark.get("strong_signals", []):
        signal_id = signal["id"]
        t_strong = parse_partial_date(signal.get("established_date"))
        if t_strong is None:
            raise ValueError(f"Missing established_date for {signal_id}")

        precursor_dates: list[date] = []
        for facts in benchmark.get("facts", {}).values():
            for fact in facts:
                for link in fact.get("precursor_of", []):
                    if link.get("signal") != signal_id:
                        continue
                    event_date = parse_partial_date(fact.get("event_date"))
                    if event_date is not None:
                        precursor_dates.append(event_date)
                    else:
                        lead_days = link.get("lead_time_days")
                        if lead_days is not None:
                            precursor_dates.append(
                                t_strong - timedelta(days=int(lead_days))
                            )

        if not precursor_dates:
            raise ValueError(f"No precursor facts found for {signal_id}")

        t_weak = min(precursor_dates)
        window_years = (t_weak.year, t_strong.year)
        signals_meta[signal_id] = {
            "id": signal_id,
            "name": signal.get("name", signal_id),
            "t_strong": t_strong,
            "t_weak": t_weak,
            "window_years": window_years,
        }

    _print_window_sanity(signals_meta)
    return signals_meta


def _print_window_sanity(signals_meta: dict[str, dict]) -> None:
    print("Computed detection windows:")
    for signal_id in sorted(signals_meta.keys()):
        meta = signals_meta[signal_id]
        computed = meta["window_years"]
        expected = EXPECTED_WINDOWS.get(signal_id)
        t_weak = meta["t_weak"].isoformat()
        t_strong = meta["t_strong"].isoformat()
        status = "ok"
        if expected is not None and computed != expected:
            status = "MISMATCH"
        expected_text = (
            f"[{expected[0]}, {expected[1]}]" if expected is not None else "n/a"
        )
        print(
            f"  {signal_id:<22} {t_weak} -> {t_strong}  "
            f"years {list(computed)}  expected {expected_text}  {status}"
        )


def load_anchors(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        print(f"ERROR: matching spec not found: {path}", file=sys.stderr)
        print("Cannot score without a rubric. Exiting.", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    anchors: dict[str, set[str]] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, list):
            continue
        words: set[str] = set()
        for entry in value:
            for token in str(entry).lower().split():
                token = token.strip()
                if token:
                    words.add(token)
        anchors[key] = words
    return anchors


def richness_guard(
    anchors: dict[str, set[str]],
    k_levels: list[int],
) -> dict[int, set[str]]:
    unreachable: dict[int, set[str]] = {k: set() for k in k_levels}
    for signal_id, anchor_set in anchors.items():
        for k in k_levels:
            if len(anchor_set) < k:
                unreachable[k].add(signal_id)
    return unreachable


def load_detections(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing detection file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    detections: list[dict] = []
    for idx, obj in enumerate(raw):
        description = ""
        for field in DESC_FIELDS:
            value = obj.get(field)
            if value is not None and str(value).strip():
                description = str(value).strip()
                break
        detections.append(
            {
                "description": description,
                "year": int(obj["year"]),
                "_orig_index": idx,
            }
        )
    return detections


def normalize(text: str) -> set[str]:
    tokens = _TOKEN_RE.split(text.lower())
    return {token for token in tokens if token}


def shared_count(desc_words: set[str], anchor_set: set[str]) -> int:
    return len(desc_words & anchor_set)


def match_at_k(
    detections: list[dict],
    signals_meta: dict[str, dict],
    anchors: dict[str, set[str]],
    k: int,
) -> dict[int, set[str]]:
    matches: dict[int, set[str]] = {}
    for det in detections:
        desc_words = normalize(det["description"])
        matched: set[str] = set()
        for signal_id, meta in sorted(signals_meta.items()):
            anchor_set = anchors.get(signal_id, set())
            if shared_count(desc_words, anchor_set) < k:
                continue
            year_low, year_high = meta["window_years"]
            if year_low <= det["year"] <= year_high:
                matched.add(signal_id)
        matches[det["_orig_index"]] = matched
    return matches


def _truncate(text: str, max_len: int = 120) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_event_matches(
    matches_at_k: dict[int, set[str]],
    detections: list[dict],
    signals_meta: dict[str, dict],
    anchors: dict[str, set[str]],
) -> list[dict]:
    det_by_index = {det["_orig_index"]: det for det in detections}
    event_matches: list[dict] = []

    for signal_id in sorted(signals_meta.keys()):
        meta = signals_meta[signal_id]
        anchor_set = anchors.get(signal_id, set())
        matching_detections: list[dict] = []

        for idx, matched_signals in sorted(matches_at_k.items()):
            if signal_id not in matched_signals:
                continue
            det = det_by_index[idx]
            desc_words = normalize(det["description"])
            matching_detections.append(
                {
                    "detection_index": idx,
                    "year": det["year"],
                    "description": det["description"],
                    "shared_anchors": sorted(desc_words & anchor_set),
                }
            )

        if matching_detections:
            event_matches.append(
                {
                    "signal_id": signal_id,
                    "signal_name": meta["name"],
                    "window_years": list(meta["window_years"]),
                    "detections": matching_detections,
                }
            )

    return event_matches


def load_overrides(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_overrides(
    matches: dict[int, set[str]],
    overrides_for_method: dict | None,
) -> dict[int, set[str]]:
    if not overrides_for_method:
        return matches

    updated = {idx: set(signals) for idx, signals in matches.items()}

    for idx, signal_id in overrides_for_method.get("force", []):
        idx = int(idx)
        if idx not in updated:
            updated[idx] = set()
        updated[idx].add(str(signal_id))

    for idx, signal_id in overrides_for_method.get("ignore", []):
        idx = int(idx)
        signal_id = str(signal_id)
        if idx in updated:
            updated[idx].discard(signal_id)

    return updated


def compute_metrics(
    matches_at_k: dict[int, set[str]],
    n_detections: int,
) -> dict:
    true_positive_indices = [
        idx for idx, matched in matches_at_k.items() if matched
    ]
    tp = len(true_positive_indices)
    events_covered: set[str] = set()
    for matched in matches_at_k.values():
        events_covered.update(matched)

    precision = tp / n_detections if n_detections else 0.0
    recall = len(events_covered) / N_EVENTS if N_EVENTS else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detections_total": n_detections,
        "true_positives": tp,
        "events_covered": len(events_covered),
        "events_covered_ids": sorted(events_covered),
        "true_positive_indices": true_positive_indices,
    }


def compute_lead_time(
    matches_at_k: dict[int, set[str]],
    detections: list[dict],
    signals_meta: dict[str, dict],
    covered_signal_ids: set[str],
) -> tuple[list[dict], float, float]:
    det_by_index = {det["_orig_index"]: det for det in detections}
    per_event: list[dict] = []
    lead_days_list: list[int] = []
    lead_years_list: list[int] = []

    for signal_id in sorted(signals_meta.keys()):
        meta = signals_meta[signal_id]
        covered = signal_id in covered_signal_ids
        det_year: int | None = None
        lead_days: int | None = None
        lead_years: int | None = None

        if covered:
            matching_years = [
                det_by_index[idx]["year"]
                for idx, matched in matches_at_k.items()
                if signal_id in matched
            ]
            det_year = min(matching_years)
            if LEAD_DATE_CONVENTION == "year_end":
                det_date = date(det_year, 12, 31)
            else:
                raise ValueError(f"Unsupported LEAD_DATE_CONVENTION: {LEAD_DATE_CONVENTION}")
            lead_days = (meta["t_strong"] - det_date).days
            lead_years = meta["t_strong"].year - det_year
            lead_days_list.append(lead_days)
            lead_years_list.append(lead_years)

        per_event.append(
            {
                "id": signal_id,
                "covered": covered,
                "det_year": det_year,
                "lead_days": lead_days,
                "lead_years": lead_years,
            }
        )

    mean_lead_days = (
        sum(lead_days_list) / len(lead_days_list) if lead_days_list else 0.0
    )
    mean_lead_years = (
        sum(lead_years_list) / len(lead_years_list) if lead_years_list else 0.0
    )
    return per_event, mean_lead_days, mean_lead_years


def score_file(
    method: str,
    path: Path,
    signals_meta: dict[str, dict],
    anchors: dict[str, set[str]],
    overrides: dict,
) -> dict[str, dict]:
    detections = load_detections(path)
    n_detections = len(detections)
    method_overrides = overrides.get(method, {})

    results: dict[str, dict] = {}
    for k in K_LEVELS:
        matches = match_at_k(detections, signals_meta, anchors, k)
        matches = apply_overrides(matches, method_overrides)

        metrics = compute_metrics(matches, n_detections)
        covered_ids = set(metrics.pop("events_covered_ids"))
        true_positive_indices = metrics.pop("true_positive_indices")

        per_event, mean_lead_days, mean_lead_years = compute_lead_time(
            matches,
            detections,
            signals_meta,
            covered_ids,
        )

        det_by_index = {det["_orig_index"]: det for det in detections}
        matched_detections = [
            {
                "detection_index": idx,
                "description": det_by_index[idx]["description"],
                "year": det_by_index[idx]["year"],
                "matched_signals": sorted(matches[idx]),
            }
            for idx in sorted(true_positive_indices)
        ]
        event_matches = build_event_matches(
            matches,
            detections,
            signals_meta,
            anchors,
        )

        results[f"k={k}"] = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "detections_total": metrics["detections_total"],
            "true_positives": metrics["true_positives"],
            "events_covered": metrics["events_covered"],
            "mean_lead_days": mean_lead_days,
            "mean_lead_years": mean_lead_years,
            "per_event": per_event,
            "matched_detections": matched_detections,
            "event_matches": event_matches,
        }

    return results


def save(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")


def print_tables(results_by_method: dict[str, dict[str, dict]], k_levels: list[int]) -> None:
    header = (
        f"{'method':<10} | {'detections':>10} | {'TP':>4} | {'precision':>9} | "
        f"{'recall':>6} | {'f1':>6} | {'events_covered':>14} | {'mean_lead_yrs':>13}"
    )
    for k in k_levels:
        print(f"\n--- k = {k} ---")
        print(header)
        print("-" * len(header))
        for method in sorted(results_by_method.keys()):
            row = results_by_method[method][f"k={k}"]
            print(
                f"{method:<10} | {row['detections_total']:>10} | "
                f"{row['true_positives']:>4} | {row['precision']:>9.3f} | "
                f"{row['recall']:>6.3f} | {row['f1']:>6.3f} | "
                f"{row['events_covered']:>14} | {row['mean_lead_years']:>13.2f}"
            )


def print_unreachable(
    unreachable_at: dict[int, set[str]],
    anchors: dict[str, set[str]],
    k_levels: list[int],
) -> None:
    print("\nUnreachable signals (anchor-richness guard):")
    for k in k_levels:
        unreachable = sorted(unreachable_at[k])
        if not unreachable:
            print(f"  k={k}: none")
            continue
        parts = [
            f"{signal_id} ({len(anchors.get(signal_id, set()))} anchors)"
            for signal_id in unreachable
        ]
        print(f"  k={k} unreachable: {', '.join(parts)}")


def print_matched_events(
    results_by_method: dict[str, dict[str, dict]],
    k_levels: list[int],
) -> None:
    print("\nMatched events (signal -> detection(s)):")
    for k in k_levels:
        print(f"\n--- k = {k} ---")
        for method in sorted(results_by_method.keys()):
            row = results_by_method[method][f"k={k}"]
            event_matches = row.get("event_matches", [])
            print(f"\n  [{method}]")
            if not event_matches:
                print("    (no events matched)")
                continue
            for event in event_matches:
                window = event["window_years"]
                print(
                    f"    Signal: {event['signal_id']} — {event['signal_name']} "
                    f"(window {window[0]}-{window[1]})"
                )
                for det in event["detections"]:
                    anchors_text = ", ".join(det["shared_anchors"]) or "(none)"
                    print(
                        f"      Detection [{det['detection_index']}] "
                        f"year={det['year']}  anchors=[{anchors_text}]"
                    )
                    print(f"        {_truncate(det['description'])}")


def print_stability(
    results_by_method: dict[str, dict[str, dict]],
    k_levels: list[int],
) -> None:
    rankings: dict[int, list[tuple[str, float]]] = {}
    for k in k_levels:
        scored = [
            (method, results_by_method[method][f"k={k}"]["f1"])
            for method in results_by_method
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        rankings[k] = scored

    rank_orders = {
        k: [method for method, _ in rankings[k]] for k in k_levels
    }
    first_order = rank_orders[k_levels[0]]
    stable = all(rank_orders[k] == first_order for k in k_levels[1:])

    print("\nRanking stability:")
    if stable:
        order_text = " > ".join(first_order)
        ks = ", ".join(str(k) for k in k_levels)
        print(f"  Ranking by F1 STABLE across k∈{{{ks}}}: {order_text}")
    else:
        print("  Ranking CHANGES with k:")
        for k in k_levels:
            order_text = " > ".join(rank_orders[k])
            print(f"    k={k}: {order_text}")


def main() -> None:
    print("Loading benchmark:", BENCHMARK_JSON)
    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    signals_meta = compute_windows(benchmark)
    anchors = load_anchors(ANCHORS_JSON)
    unreachable_at = richness_guard(anchors, K_LEVELS)
    overrides = load_overrides(OVERRIDES_JSON)

    results_by_method: dict[str, dict[str, dict]] = {}
    for method, detection_path in DETECTION_FILES:
        print(f"\nScoring {method}: {detection_path.name}")
        results = score_file(
            method,
            detection_path,
            signals_meta,
            anchors,
            overrides,
        )
        output_path = OUTPUT_DIR / f"results_{method}.json"
        save(results, output_path)
        print(f"  Wrote {output_path}")
        results_by_method[method] = results

    print_tables(results_by_method, K_LEVELS)
    print_matched_events(results_by_method, K_LEVELS)
    print_unreachable(unreachable_at, anchors, K_LEVELS)
    print_stability(results_by_method, K_LEVELS)


if __name__ == "__main__":
    print("=" * 50)
    print("  WIKI-OPENAI WEAK-SIGNAL BENCHMARK SCORER")
    print("=" * 50)
    main()
