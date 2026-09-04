#!/usr/bin/env python3
"""
Yoon (2012) keyword-based weak-signal detector on the OpenAI weak-signal benchmark.

Reads yearly fact texts from openai_weak_signal_benchmark.json, runs expanding-window
Area-A detection on Emergence and Issue maps, and writes detections_yoon.json.
Does not score detections (a separate shared scorer handles that).

Usage:
    pyenv activate venv-itext2kg-update
    python evaluation/c_unseen/detectors/keyword_based_yoon.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

for _p in Path(__file__).resolve().parents:
    if (_p / ".git").exists():
        project_root = _p
        break
else:
    raise RuntimeError("Could not locate repository root")
sys.path.insert(0, str(project_root))

from evaluation.c_unseen.paths import BENCHMARK, OUTPUTS  # noqa: E402

# ==========================
# User-configurable globals
# ==========================
INPUT_JSON: Path = BENCHMARK
OUTPUT_JSON: Path = OUTPUTS / "detections_yoon.json"

TW: float = 0.05
GROWTH_PERCENTILE: float = 70
MIN_DOC_FREQ: int = 2
NGRAM_RANGE: tuple[int, int] = (1, 2)
STOPWORDS: str = "english"
LEMMATIZE: bool = False  # optional in spec; left off for determinism / simplicity

TOKEN_PATTERN: str = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"


def load_data(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["facts"]


def build_periods(facts_by_year: dict[str, list[dict]]) -> list[tuple[int, list[str]]]:
    periods: list[tuple[int, list[str]]] = []
    for year_str in sorted(facts_by_year.keys(), key=int):
        texts = [
            str(fact.get("text", "")).strip()
            for fact in facts_by_year[year_str]
            if str(fact.get("text", "")).strip()
        ]
        periods.append((int(year_str), texts))
    return periods


def _make_vectorizer(vocabulary: list[str] | None = None) -> CountVectorizer:
    kwargs: dict = {
        "stop_words": STOPWORDS,
        "ngram_range": NGRAM_RANGE,
        "lowercase": True,
        "token_pattern": TOKEN_PATTERN,
    }
    if vocabulary is None:
        kwargs["min_df"] = MIN_DOC_FREQ
    else:
        kwargs["vocabulary"] = vocabulary
    return CountVectorizer(**kwargs)


def extract_keywords(period_texts: list[list[str]]) -> list[str]:
    corpus = [" ".join(texts) for texts in period_texts]
    if not any(corpus):
        return []
    vectorizer = _make_vectorizer()
    vectorizer.fit(corpus)
    return sorted(vectorizer.get_feature_names_out())


def count_tf_df(
    periods: list[tuple[int, list[str]]],
    vocab: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_periods = len(periods)
    n_keywords = len(vocab)
    tf = np.zeros((n_keywords, n_periods), dtype=np.float64)
    df = np.zeros((n_keywords, n_periods), dtype=np.float64)
    nn = np.zeros(n_periods, dtype=np.float64)

    if n_keywords == 0:
        return tf, df, nn

    vectorizer = _make_vectorizer(vocabulary=vocab)
    for j, (_, texts) in enumerate(periods):
        nn[j] = len(texts)
        if not texts:
            continue
        matrix = vectorizer.transform(texts)
        tf[:, j] = np.asarray(matrix.sum(axis=0)).ravel()
        df[:, j] = np.asarray((matrix > 0).sum(axis=0)).ravel()

    return tf, df, nn


def compute_dov_dod(
    tf: np.ndarray,
    df: np.ndarray,
    nn: np.ndarray,
    tw: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_keywords, n_periods = tf.shape
    dov = np.zeros_like(tf)
    dod = np.zeros_like(df)

    for j in range(n_periods):
        if nn[j] <= 0:
            continue
        period_weight = 1.0 - tw * (n_periods - (j + 1))
        dov[:, j] = (tf[:, j] / nn[j]) * period_weight
        dod[:, j] = (df[:, j] / nn[j]) * period_weight

    return dov, dod


def growth(series: np.ndarray) -> float:
    positive = np.where(series > 0)[0]
    if len(positive) <= 1:
        return 0.0
    p = int(positive[0])
    q = int(positive[-1])
    if p == q:
        return 0.0
    value_p = series[p]
    value_q = series[q]
    if value_p <= 0:
        return 0.0
    return float((value_q / value_p) ** (1.0 / (q - p)) - 1.0)


def map_coordinates(
    tf: np.ndarray,
    df: np.ndarray,
    dov: np.ndarray,
    dod: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_keywords = tf.shape[0]
    x_em = tf.mean(axis=1)
    x_is = df.mean(axis=1)
    y_em = np.array([growth(dov[i]) for i in range(n_keywords)], dtype=np.float64)
    y_is = np.array([growth(dod[i]) for i in range(n_keywords)], dtype=np.float64)
    return x_em, y_em, x_is, y_is


def flag_area_A(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=bool)
    growth_threshold = np.percentile(y, GROWTH_PERCENTILE)
    rare_threshold = x.mean()
    return (y >= growth_threshold) & (x < rare_threshold)


def _run_window(periods: list[tuple[int, list[str]]]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    period_texts = [texts for _, texts in periods]
    vocab = extract_keywords(period_texts)
    if not vocab:
        empty = np.array([], dtype=bool)
        empty_f = np.array([], dtype=np.float64)
        return vocab, empty, empty, empty_f, empty_f, empty_f, empty_f

    tf, df, nn = count_tf_df(periods, vocab)
    dov, dod = compute_dov_dod(tf, df, nn, TW)
    x_em, y_em, x_is, y_is = map_coordinates(tf, df, dov, dod)
    emergence_flags = flag_area_A(x_em, y_em)
    issue_flags = flag_area_A(x_is, y_is)
    return vocab, emergence_flags, issue_flags, x_em, y_em, x_is, y_is


def expanding_window_detect(
    periods: list[tuple[int, list[str]]],
) -> tuple[list[dict], int, dict[int, int]]:
    if len(periods) < 2:
        return [], 0, {}

    detections: dict[str, dict] = {}
    flagged_per_year: dict[int, int] = {}
    full_window_vocab_size = 0

    for end_idx in range(1, len(periods)):
        window = periods[: end_idx + 1]
        year = window[-1][0]

        (
            vocab,
            emergence_flags,
            issue_flags,
            x_em,
            y_em,
            x_is,
            y_is,
        ) = _run_window(window)

        if end_idx == len(periods) - 1:
            full_window_vocab_size = len(vocab)

        if not vocab:
            flagged_per_year[year] = 0
            continue

        newly_flagged = 0
        for i, keyword in enumerate(vocab):
            maps: list[str] = []
            if emergence_flags[i]:
                maps.append("emergence")
            if issue_flags[i]:
                maps.append("issue")
            if not maps:
                continue
            if keyword in detections:
                continue

            detections[keyword] = {
                "keyword": keyword,
                "year": year,
                "maps": maps,
                "x_emergence": float(x_em[i]),
                "y_emergence": float(y_em[i]),
                "x_issue": float(x_is[i]),
                "y_issue": float(y_is[i]),
            }
            newly_flagged += 1

        flagged_per_year[year] = newly_flagged

    results = sorted(detections.values(), key=lambda row: (row["year"], row["keyword"]))
    return results, full_window_vocab_size, flagged_per_year


def save(detections: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)
        f.write("\n")


def _print_summary(
    candidate_keywords: int,
    flagged_per_year: dict[int, int],
    detections: list[dict],
) -> None:
    emergence_only = sum(1 for d in detections if d["maps"] == ["emergence"])
    issue_only = sum(1 for d in detections if d["maps"] == ["issue"])
    both_maps = sum(1 for d in detections if len(d["maps"]) == 2)

    print(f"Candidate keywords (full window): {candidate_keywords}")
    print("Flagged per year (new detections):")
    for year in sorted(flagged_per_year):
        print(f"  {year}: {flagged_per_year[year]}")
    print(f"Total detections: {len(detections)}")
    print(f"  emergence-only: {emergence_only}")
    print(f"  issue-only: {issue_only}")
    print(f"  both maps (strict): {both_maps}")


def main() -> None:
    print("Loading benchmark:", INPUT_JSON)
    facts = load_data(INPUT_JSON)
    periods = build_periods(facts)
    print(f"Periods: {periods[0][0]}..{periods[-1][0]} ({len(periods)} years)")

    detections, candidate_keywords, flagged_per_year = expanding_window_detect(periods)
    save(detections, OUTPUT_JSON)

    print("Wrote:", OUTPUT_JSON)
    _print_summary(candidate_keywords, flagged_per_year, detections)


if __name__ == "__main__":
    print("=" * 50)
    print("  YOON (2012) KEYWORD-BASED WEAK-SIGNAL DETECTOR")
    print("=" * 50)
    main()
