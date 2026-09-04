#!/usr/bin/env python3
"""
BERTrend (Boutaleb et al., 2024) topic-modeling weak-signal detector.

Runs automatic topic discovery on the OpenAI weak-signal benchmark, tracks topics
across yearly slices, classifies weak/strong/noise signals, and writes
detections_bertrend.json. Does not score detections (a separate shared scorer
handles that).

Usage:
    pyenv activate venv-itext2kg-update
    pip install umap-learn   # one-time if missing
    python evaluation/c_unseen/detectors/bertrend_topic_modeling.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
OUTPUT_JSON: Path = OUTPUTS / "detections_bertrend.json"

EMBEDDING_MODEL: str = "text-embedding-3-large"
MERGE_SIM_THRESHOLD: float = 0.6
MIN_CLUSTER_SIZE: int = 2
MIN_SAMPLES: int = 1
UMAP_N_COMPONENTS: int = 5
UMAP_N_NEIGHBORS: int = 15
DECAY_LAMBDA: float = 0.1
RETRO_WINDOW_YEARS: int | None = None
P_LOW: int = 10
P_MID: int = 50
DETECTION_LABELS: list[str] = ["weak"]
NEW_TOPIC_IN_MIDBAND_IS_WEAK: bool = True
TOP_WORDS_PER_TOPIC: int = 10
NGRAM_RANGE: tuple[int, int] = (1, 2)
STOPWORDS: str = "english"
RANDOM_STATE: int = 42
TOKEN_PATTERN: str = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
HDBSCAN_METRIC: str = "euclidean"


@dataclass
class SliceTopic:
    description: str
    doc_indices: list[int]
    centroid_embedding: np.ndarray


@dataclass
class GlobalTopic:
    topic_id: int
    description: str
    description_embedding: np.ndarray
    first_year: int
    last_update_year: int | None = None
    docs_added_this_year: int = 0
    popularity_by_year: dict[int, float] = field(default_factory=dict)
    label_by_year: dict[int, str] = field(default_factory=dict)


def _resolve_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        from api_keys import openai_api_key

        return openai_api_key
    except ImportError as exc:
        raise RuntimeError(
            "OPENAI_API_KEY is not set and api_keys.py is unavailable."
        ) from exc


def load_data(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["facts"]


def build_year_slices(
    facts_by_year: dict[str, list[dict]],
) -> list[tuple[int, list[str]]]:
    slices: list[tuple[int, list[str]]] = []
    for year_str in sorted(facts_by_year.keys(), key=int):
        texts = [
            str(fact.get("text", "")).strip()
            for fact in facts_by_year[year_str]
            if str(fact.get("text", "")).strip()
        ]
        slices.append((int(year_str), texts))
    return slices


def embed_all_facts(
    slices: list[tuple[int, list[str]]],
    embeddings_model: OpenAIEmbeddings,
) -> dict[int, np.ndarray]:
    all_texts: list[str] = []
    year_doc_counts: list[tuple[int, int]] = []
    for year, texts in slices:
        year_doc_counts.append((year, len(texts)))
        all_texts.extend(texts)

    if not all_texts:
        return {year: np.zeros((0, 0), dtype=np.float64) for year, _ in slices}

    vectors = embeddings_model.embed_documents(all_texts)
    matrix = np.asarray(vectors, dtype=np.float64)

    by_year: dict[int, np.ndarray] = {}
    offset = 0
    for year, count in year_doc_counts:
        if count == 0:
            by_year[year] = np.zeros((0, matrix.shape[1]), dtype=np.float64)
        else:
            by_year[year] = matrix[offset : offset + count]
            offset += count
    return by_year


def _make_vectorizer() -> CountVectorizer:
    return CountVectorizer(
        stop_words=STOPWORDS,
        ngram_range=NGRAM_RANGE,
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
    )


def _top_terms_from_docs(docs: list[str], top_n: int) -> str:
    if not docs:
        return ""
    vectorizer = _make_vectorizer()
    matrix = vectorizer.fit_transform(docs)
    if matrix.shape[1] == 0:
        return docs[0][:80]
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    feature_names = vectorizer.get_feature_names_out()
    top_idx = np.argsort(scores)[::-1][:top_n]
    return " ".join(feature_names[i] for i in top_idx if scores[i] > 0)


def _ctfidf_descriptions(
    texts: list[str],
    cluster_to_indices: dict[int, list[int]],
    top_n: int,
) -> dict[int, str]:
    if not texts or not cluster_to_indices:
        return {}

    vectorizer = _make_vectorizer()
    doc_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    n_features = doc_matrix.shape[1]
    if n_features == 0:
        return {
            cluster_id: _top_terms_from_docs([texts[i] for i in indices], top_n)
            for cluster_id, indices in cluster_to_indices.items()
        }

    class_ids = sorted(cluster_to_indices.keys())
    class_term_counts = np.zeros((len(class_ids), n_features), dtype=np.float64)
    class_sizes = np.zeros(len(class_ids), dtype=np.float64)

    for row, cluster_id in enumerate(class_ids):
        indices = cluster_to_indices[cluster_id]
        sub = doc_matrix[indices]
        counts = np.asarray(sub.sum(axis=0)).ravel()
        class_term_counts[row] = counts
        class_sizes[row] = counts.sum()

    descriptions: dict[int, str] = {}
    n_classes = len(class_ids)
    for row, cluster_id in enumerate(class_ids):
        if class_sizes[row] <= 0:
            descriptions[cluster_id] = _top_terms_from_docs(
                [texts[i] for i in cluster_to_indices[cluster_id]],
                top_n,
            )
            continue

        tf = class_term_counts[row] / class_sizes[row]
        df = (class_term_counts > 0).sum(axis=0)
        idf = np.log(1.0 + (n_classes / np.maximum(df, 1)))
        scores = tf * idf
        top_idx = np.argsort(scores)[::-1][:top_n]
        terms = [feature_names[i] for i in top_idx if scores[i] > 0]
        if not terms:
            terms = [feature_names[i] for i in top_idx[:top_n]]
        descriptions[cluster_id] = " ".join(terms)

    return descriptions


def _fit_hdbscan(reduced: np.ndarray) -> np.ndarray:
    try:
        from fast_hdbscan import HDBSCAN
    except ImportError as exc:
        raise ImportError(
            "fast_hdbscan is required. Install with: pip install fast_hdbscan"
        ) from exc

    clusterer = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
    )
    return clusterer.fit_predict(reduced.astype(np.float64))


def _singleton_topics(texts: list[str], embeddings: np.ndarray) -> list[SliceTopic]:
    topics: list[SliceTopic] = []
    for idx, text in enumerate(texts):
        topics.append(
            SliceTopic(
                description=_top_terms_from_docs([text], TOP_WORDS_PER_TOPIC),
                doc_indices=[idx],
                centroid_embedding=embeddings[idx],
            )
        )
    return topics


def fit_slice_topics(
    year: int,
    texts: list[str],
    embeddings: np.ndarray,
) -> list[SliceTopic]:
    del year  # kept for API symmetry / future logging
    n_docs = len(texts)
    if n_docs == 0:
        return []

    if n_docs < 3:
        return _singleton_topics(texts, embeddings)

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required. Install with: pip install umap-learn"
        ) from exc

    n_neighbors = min(UMAP_N_NEIGHBORS, n_docs - 1)
    n_components = min(UMAP_N_COMPONENTS, max(2, n_docs - 2))

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    reduced = reducer.fit_transform(embeddings)

    labels = _fit_hdbscan(reduced)
    cluster_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        cluster_to_indices.setdefault(int(label), []).append(idx)

    if not cluster_to_indices:
        return []

    descriptions = _ctfidf_descriptions(texts, cluster_to_indices, TOP_WORDS_PER_TOPIC)
    topics: list[SliceTopic] = []
    for cluster_id in sorted(cluster_to_indices.keys()):
        indices = cluster_to_indices[cluster_id]
        centroid = embeddings[indices].mean(axis=0)
        topics.append(
            SliceTopic(
                description=descriptions.get(
                    cluster_id,
                    _top_terms_from_docs([texts[i] for i in indices], TOP_WORDS_PER_TOPIC),
                ),
                doc_indices=indices,
                centroid_embedding=centroid,
            )
        )
    return topics


def merge_topics(
    global_topics: list[GlobalTopic],
    slice_topics: list[SliceTopic],
    year: int,
    embed_text,
    next_topic_id: int,
) -> tuple[int, int, int]:
    n_merged = 0
    n_new = 0

    for slice_topic in slice_topics:
        description_embedding = embed_text(slice_topic.description)
        n_docs = len(slice_topic.doc_indices)

        best_idx: int | None = None
        best_sim = -1.0
        if global_topics:
            existing_embeddings = np.vstack(
                [topic.description_embedding for topic in global_topics]
            )
            sims = cosine_similarity(
                description_embedding.reshape(1, -1),
                existing_embeddings,
            ).ravel()
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

        if best_idx is not None and best_sim >= MERGE_SIM_THRESHOLD:
            topic = global_topics[best_idx]
            topic.docs_added_this_year += n_docs
            n_merged += 1
        else:
            global_topics.append(
                GlobalTopic(
                    topic_id=next_topic_id,
                    description=slice_topic.description,
                    description_embedding=description_embedding,
                    first_year=year,
                    docs_added_this_year=n_docs,
                )
            )
            next_topic_id += 1
            n_new += 1

    return n_merged, n_new, next_topic_id


def update_popularity(global_topics: list[GlobalTopic], year: int) -> None:
    for topic in global_topics:
        if topic.first_year > year:
            continue

        if topic.docs_added_this_year > 0:
            if topic.last_update_year is None:
                topic.popularity_by_year[year] = float(topic.docs_added_this_year)
            else:
                prev_pop = topic.popularity_by_year.get(topic.last_update_year, 0.0)
                topic.popularity_by_year[year] = prev_pop + float(
                    topic.docs_added_this_year
                )
            topic.last_update_year = year
        elif topic.last_update_year is not None:
            dt = year - topic.last_update_year
            prev_pop = topic.popularity_by_year.get(topic.last_update_year, 0.0)
            topic.popularity_by_year[year] = prev_pop * math.exp(
                -DECAY_LAMBDA * (dt**2)
            )
        else:
            topic.popularity_by_year[year] = 0.0

        topic.docs_added_this_year = 0


def _window_years(year: int, all_years: list[int]) -> list[int]:
    if RETRO_WINDOW_YEARS is None:
        return [y for y in all_years if y <= year]
    start = year - RETRO_WINDOW_YEARS
    return [y for y in all_years if start <= y <= year]


def _popularity_slope(pop_by_year: dict[int, float], window_years: list[int]) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for y in window_years:
        if y in pop_by_year:
            xs.append(float(y))
            ys.append(float(pop_by_year[y]))
    if len(xs) <= 1:
        return float("nan")
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if np.allclose(x_arr, x_arr[0]):
        return 0.0
    slope, _ = np.polyfit(x_arr, y_arr, 1)
    return float(slope)


def classify_signals(
    global_topics: list[GlobalTopic],
    year: int,
    all_years: list[int],
) -> dict[int, str]:
    window = _window_years(year, all_years)
    pool_values: list[float] = []
    for topic in global_topics:
        if topic.first_year > year:
            continue
        for y in window:
            if y in topic.popularity_by_year:
                pool_values.append(topic.popularity_by_year[y])

    if not pool_values:
        return {}

    p_low = float(np.percentile(pool_values, P_LOW))
    p_mid = float(np.percentile(pool_values, P_MID))

    labels: dict[int, str] = {}
    for topic in global_topics:
        if topic.first_year > year:
            continue

        p = topic.popularity_by_year.get(year, 0.0)
        if p < p_low:
            labels[topic.topic_id] = "noise"
            continue

        if p > p_mid:
            labels[topic.topic_id] = "strong"
            continue

        slope = _popularity_slope(topic.popularity_by_year, window)
        if math.isnan(slope):
            labels[topic.topic_id] = (
                "weak" if NEW_TOPIC_IN_MIDBAND_IS_WEAK else "noise"
            )
        elif slope > 0:
            labels[topic.topic_id] = "weak"
        else:
            labels[topic.topic_id] = "noise"

    return labels


def extract_detections(global_topics: list[GlobalTopic]) -> list[dict]:
    detections: list[dict] = []
    for topic in global_topics:
        detection_year: int | None = None
        detection_label: str | None = None
        for year in sorted(topic.label_by_year.keys()):
            label = topic.label_by_year[year]
            if label in DETECTION_LABELS:
                detection_year = year
                detection_label = label
                break
        if detection_year is None or detection_label is None:
            continue

        detections.append(
            {
                "topic_id": topic.topic_id,
                "description": topic.description,
                "year": detection_year,
                "signal_at_detection": detection_label,
                "popularity_by_year": {
                    str(y): float(v)
                    for y, v in sorted(topic.popularity_by_year.items())
                },
                "label_by_year": {
                    str(y): label
                    for y, label in sorted(topic.label_by_year.items())
                },
            }
        )

    detections.sort(key=lambda row: (row["year"], row["topic_id"]))
    return detections


def save(detections: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)
        f.write("\n")


def _print_summary(
    slices: list[tuple[int, list[str]]],
    yearly_stats: list[dict],
    detections: list[dict],
) -> None:
    total_docs = sum(len(texts) for _, texts in slices)
    print(f"Total slices: {len(slices)} | total documents: {total_docs}")
    print("Per year:")
    for row in yearly_stats:
        print(
            f"  {row['year']}: docs={row['docs']} topics_fit={row['topics_fit']} "
            f"merged={row['merged']} new={row['new']} weak={row['weak']} "
            f"strong={row['strong']}"
        )

    weak_count = sum(1 for d in detections if d["signal_at_detection"] == "weak")
    strong_count = sum(1 for d in detections if d["signal_at_detection"] == "strong")
    print(f"Total detections: {len(detections)}")
    print(f"  weak: {weak_count}")
    print(f"  strong: {strong_count}")


def main() -> None:
    np.random.seed(RANDOM_STATE)

    print("Loading benchmark:", INPUT_JSON)
    facts = load_data(INPUT_JSON)
    slices = build_year_slices(facts)
    all_years = [year for year, _ in slices]

    api_key = _resolve_openai_api_key()
    embeddings_model = OpenAIEmbeddings(
        api_key=api_key,
        model=EMBEDDING_MODEL,
    )

    print("Embedding all facts with", EMBEDDING_MODEL)
    embeddings_by_year = embed_all_facts(slices, embeddings_model)

    embedding_cache: dict[str, np.ndarray] = {}

    def embed_text(text: str) -> np.ndarray:
        if text not in embedding_cache:
            vec = embeddings_model.embed_documents([text])[0]
            embedding_cache[text] = np.asarray(vec, dtype=np.float64)
        return embedding_cache[text]

    global_topics: list[GlobalTopic] = []
    next_topic_id = 0
    yearly_stats: list[dict] = []

    for year, texts in slices:
        slice_embeddings = embeddings_by_year[year]
        slice_topics = fit_slice_topics(year, texts, slice_embeddings)
        n_merged, n_new, next_topic_id = merge_topics(
            global_topics,
            slice_topics,
            year,
            embed_text,
            next_topic_id,
        )
        update_popularity(global_topics, year)
        labels = classify_signals(global_topics, year, all_years)
        for topic in global_topics:
            if topic.topic_id in labels:
                topic.label_by_year[year] = labels[topic.topic_id]

        n_weak = sum(1 for label in labels.values() if label == "weak")
        n_strong = sum(1 for label in labels.values() if label == "strong")
        yearly_stats.append(
            {
                "year": year,
                "docs": len(texts),
                "topics_fit": len(slice_topics),
                "merged": n_merged,
                "new": n_new,
                "weak": n_weak,
                "strong": n_strong,
            }
        )

    detections = extract_detections(global_topics)
    save(detections, OUTPUT_JSON)

    print("Wrote:", OUTPUT_JSON)
    _print_summary(slices, yearly_stats, detections)


if __name__ == "__main__":
    print("=" * 50)
    print("  BERTREND TOPIC-MODELING WEAK-SIGNAL DETECTOR")
    print("=" * 50)
    main()
