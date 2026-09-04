#!/usr/bin/env python3
"""
BEAM (Abou Jamra et al., 2022) graphlet-based weak-signal detector.

Builds yearly untyped graphs from OpenAI weak-signal KG snapshots, counts 2–4 node
graphlets (G0..G8), applies BEAM emergence / rareness / persistence filters, grounds
confirmed precursors to entity names, and writes detections_beam.json.

Usage:
    pyenv activate venv-itext2kg-update
    python evaluation/c_unseen/detectors/graphlet_beam.py
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import networkx as nx
import numpy as np

for _p in Path(__file__).resolve().parents:
    if (_p / ".git").exists():
        project_root = _p
        break
else:
    raise RuntimeError("Could not locate repository root")
sys.path.insert(0, str(project_root))

from itext2kg.atom.models import KnowledgeGraph  # noqa: E402
from evaluation.c_unseen.paths import OUTPUTS, SNAPSHOTS  # noqa: E402

# ==========================
# User-configurable globals
# ==========================
SNAPSHOTS_DIR: Path = SNAPSHOTS
OUTPUT_JSON: Path = OUTPUTS / "detections_beam.json"

MAX_GRAPHLET_NODES: int = 4
TOP_K: int = 5
CONFIRM_HORIZON: int = 1
GROUND_TOP_N: int = 10
SPLIT_METHOD: str = "threshold"

GRAPHLET_NAMES: list[str] = [f"G{i}" for i in range(9)]

# (n_nodes, n_edges, sorted_degree_sequence) -> graphlet id
SHAPE_FINGERPRINTS: dict[tuple[int, int, tuple[int, ...]], str] = {
    (2, 1, (1, 1)): "G0",
    (3, 2, (1, 1, 2)): "G1",
    (3, 3, (2, 2, 2)): "G2",
    (4, 3, (1, 1, 2, 2)): "G3",
    (4, 3, (1, 1, 1, 3)): "G4",
    (4, 4, (2, 2, 2, 2)): "G5",
    (4, 4, (1, 2, 2, 3)): "G6",
    (4, 5, (2, 2, 3, 3)): "G7",
    (4, 6, (3, 3, 3, 3)): "G8",
}

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_entity(name: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(name).lower().strip())


def load_snapshots(snapshots_dir: Path) -> list[tuple[int, KnowledgeGraph]]:
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"Missing snapshots dir: {snapshots_dir}")

    snapshots: list[tuple[int, KnowledgeGraph]] = []
    paths = [
        path
        for path in snapshots_dir.glob("*.pkl")
        if path.stem.isdigit()
    ]
    for path in sorted(paths, key=lambda p: int(p.stem)):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        kg = payload["knowledge_graph"]
        year = int(payload.get("year", path.stem))
        snapshots.append((year, kg))

    if not snapshots:
        raise FileNotFoundError(f"No yearly snapshots found in {snapshots_dir}")
    return snapshots


def build_year_graphs(
    snapshots: list[tuple[int, KnowledgeGraph]],
) -> list[tuple[int, nx.Graph, dict[tuple[str, str], list[str]]]]:
    year_graphs: list[tuple[int, nx.Graph, dict[tuple[str, str], list[str]]]] = []

    for year, kg in snapshots:
        graph = nx.Graph()
        edge_to_facts: dict[tuple[str, str], list[str]] = {}

        for rel in kg.relationships:
            start = _normalize_entity(rel.startEntity.name)
            end = _normalize_entity(rel.endEntity.name)
            if not start or not end or start == end:
                continue

            graph.add_edge(start, end)
            edge = tuple(sorted((start, end)))
            facts = [str(f).strip() for f in rel.properties.atomic_facts if str(f).strip()]
            if edge not in edge_to_facts:
                edge_to_facts[edge] = []
            edge_to_facts[edge].extend(facts)

        for edge in edge_to_facts:
            edge_to_facts[edge] = sorted(set(edge_to_facts[edge]))

        year_graphs.append((year, graph, edge_to_facts))

    return year_graphs


def enumerate_connected_subgraphs(graph: nx.Graph, k: int) -> Iterator[frozenset[str]]:
    if k <= 0 or graph.number_of_nodes() < k:
        return

    def extend(
        subgraph: set[str],
        candidates: set[str],
    ) -> Iterator[frozenset[str]]:
        if len(subgraph) == k:
            yield frozenset(subgraph)
            return
        if not candidates:
            return

        for node in sorted(candidates):
            new_subgraph = set(subgraph)
            new_subgraph.add(node)
            new_candidates: set[str] = set()
            for vertex in new_subgraph:
                for neighbor in graph.neighbors(vertex):
                    if neighbor not in new_subgraph and neighbor > min(new_subgraph):
                        new_candidates.add(neighbor)
            yield from extend(new_subgraph, new_candidates)

    for root in sorted(graph.nodes()):
        candidates = {n for n in graph.neighbors(root) if n > root}
        yield from extend({root}, candidates)


def classify_shape(graph: nx.Graph, nodes: frozenset[str]) -> str | None:
    sub = graph.subgraph(nodes)
    if not nx.is_connected(sub):
        return None

    n_nodes = sub.number_of_nodes()
    n_edges = sub.number_of_edges()
    degrees = tuple(sorted(dict(sub.degree()).values()))
    return SHAPE_FINGERPRINTS.get((n_nodes, n_edges, degrees))


def count_graphlets(graph: nx.Graph) -> dict[str, int]:
    counts = {name: 0 for name in GRAPHLET_NAMES}
    if graph.number_of_nodes() < 2:
        return counts

    for k in range(2, MAX_GRAPHLET_NODES + 1):
        for nodes in enumerate_connected_subgraphs(graph, k):
            shape = classify_shape(graph, nodes)
            if shape is None:
                continue
            counts[shape] += 1

    return counts


def collect_instances_for_shape(
    graph: nx.Graph,
    shape: str,
) -> list[tuple[str, ...]]:
    instances: list[tuple[str, ...]] = []
    if graph.number_of_nodes() < 2:
        return instances

    min_k = 2 if shape == "G0" else 3 if shape in {"G1", "G2"} else 4
    max_k = 2 if shape == "G0" else 3 if shape in {"G1", "G2"} else 4

    for k in range(min_k, max_k + 1):
        for nodes in enumerate_connected_subgraphs(graph, k):
            if classify_shape(graph, nodes) == shape:
                instances.append(tuple(sorted(nodes)))
    return instances


def normalize_counts(count_matrix: np.ndarray) -> np.ndarray:
    n_shapes, n_years = count_matrix.shape
    normalized = np.zeros_like(count_matrix, dtype=np.float64)
    for i in range(n_shapes):
        series = count_matrix[i]
        mean = float(series.mean())
        std = float(series.std())
        if std == 0:
            normalized[i] = 0.0
        else:
            normalized[i] = (series - mean) / std
    return normalized


def velocity_acceleration(
    normalized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_shapes, n_years = normalized.shape
    velocity = np.zeros_like(normalized)
    acceleration = np.zeros_like(normalized)

    if n_years >= 2:
        velocity[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
    if n_years >= 3:
        acceleration[:, 2:] = velocity[:, 2:] - velocity[:, 1:-1]

    return velocity, acceleration


def emergence_map_precursors(
    velocity_t: np.ndarray,
    acceleration_t: np.ndarray,
) -> dict[str, str]:
    if SPLIT_METHOD != "threshold":
        raise ValueError(f"Unsupported SPLIT_METHOD: {SPLIT_METHOD}")

    v_mean = float(velocity_t.mean())
    a_mean = float(acceleration_t.mean())
    precursors: dict[str, str] = {}

    for idx, shape in enumerate(GRAPHLET_NAMES):
        v = float(velocity_t[idx])
        a = float(acceleration_t[idx])
        high_v = v >= v_mean
        high_a = a >= a_mean

        if high_v and high_a:
            precursors[shape] = "Q2"
        elif (not high_v) and high_a:
            precursors[shape] = "Q1"
        elif high_v and (not high_a):
            precursors[shape] = "Q4"
        # Q3: low velocity and low acceleration -> noise (discard)

    return precursors


def contribution_ratios(
    counts_year_t: dict[str, int],
    counts_total: dict[str, int],
) -> tuple[dict[str, float], dict[str, float]]:
    local_total = float(sum(counts_year_t.values()))
    global_total = float(sum(counts_total.values()))

    local_ratios: dict[str, float] = {}
    global_ratios: dict[str, float] = {}
    for shape in GRAPHLET_NAMES:
        local_ratios[shape] = (
            counts_year_t.get(shape, 0) / local_total if local_total > 0 else 0.0
        )
        global_ratios[shape] = (
            counts_total.get(shape, 0) / global_total if global_total > 0 else 0.0
        )
    return local_ratios, global_ratios


def pick_top_k_rare(
    precursors: dict[str, str],
    global_ratios: dict[str, float],
) -> list[str]:
    ranked = sorted(
        precursors.keys(),
        key=lambda shape: (global_ratios.get(shape, 0.0), shape),
    )
    return ranked[:TOP_K]


def persistence_filter(
    shape: str,
    year_index: int,
    velocity: np.ndarray,
) -> bool:
    shape_idx = GRAPHLET_NAMES.index(shape)
    for offset in range(1, CONFIRM_HORIZON + 1):
        future_idx = year_index + offset
        if future_idx >= velocity.shape[1]:
            return False
        if float(velocity[shape_idx, future_idx]) > 0.0:
            return True
    return False


def ground_entities(
    instances: list[tuple[str, ...]],
) -> tuple[str, list[str]]:
    if not instances:
        return "", []

    entity_counts: Counter[str] = Counter()
    for node_tuple in instances:
        entity_counts.update(node_tuple)

    top_entities = [entity for entity, _ in entity_counts.most_common(GROUND_TOP_N)]
    description = " ".join(top_entities)
    return description, sorted(top_entities)


def _counts_matrix(
    yearly_counts: list[dict[str, int]],
    end_exclusive: int,
) -> np.ndarray:
    matrix = np.zeros((len(GRAPHLET_NAMES), end_exclusive), dtype=np.float64)
    for j in range(end_exclusive):
        for i, shape in enumerate(GRAPHLET_NAMES):
            matrix[i, j] = yearly_counts[j].get(shape, 0)
    return matrix


def extract_detections(
    year_graphs: list[tuple[int, nx.Graph, dict[tuple[str, str], list[str]]]],
) -> tuple[list[dict], list[dict], list[dict]]:
    years = [year for year, _, _ in year_graphs]
    graphs = [graph for _, graph, _ in year_graphs]

    yearly_counts: list[dict[str, int]] = []
    for year, graph, _ in year_graphs:
        print(f"  Counting graphlets for {year}...", flush=True)
        yearly_counts.append(count_graphlets(graph))

    detections: list[dict] = []
    vector_rows: list[dict] = []
    yearly_stats: list[dict] = []

    n_years = len(years)
    # Acceleration available from the 3rd snapshot (index 2); persistence needs t+1.
    for t_idx in range(2, n_years - CONFIRM_HORIZON):
        year = years[t_idx]

        # Steps 1–5: causal window up to year t.
        counts_upto_t = yearly_counts[: t_idx + 1]
        count_matrix_t = _counts_matrix(counts_upto_t, t_idx + 1)
        normalized_t = normalize_counts(count_matrix_t)
        velocity_t, acceleration_t = velocity_acceleration(normalized_t)

        precursors = emergence_map_precursors(
            velocity_t[:, t_idx],
            acceleration_t[:, t_idx],
        )

        counts_year_t = yearly_counts[t_idx]
        counts_total_upto_t = {shape: 0 for shape in GRAPHLET_NAMES}
        for counts in counts_upto_t:
            for shape in GRAPHLET_NAMES:
                counts_total_upto_t[shape] += counts.get(shape, 0)

        local_ratios, global_ratios = contribution_ratios(
            counts_year_t,
            counts_total_upto_t,
        )
        top_rare = pick_top_k_rare(precursors, global_ratios)

        # Step 6: persistence uses velocity at t+1 (window extended by one snapshot).
        counts_upto_t1 = yearly_counts[: t_idx + 1 + CONFIRM_HORIZON]
        count_matrix_t1 = _counts_matrix(counts_upto_t1, t_idx + 1 + CONFIRM_HORIZON)
        normalized_t1 = normalize_counts(count_matrix_t1)
        velocity_t1, _ = velocity_acceleration(normalized_t1)

        n_confirmed = 0
        for shape in top_rare:
            if shape not in precursors:
                continue
            if not persistence_filter(shape, t_idx, velocity_t1):
                continue

            description, instance_entities = ground_entities(
                collect_instances_for_shape(graphs[t_idx], shape)
            )
            if not description:
                continue

            detections.append(
                {
                    "graphlet": shape,
                    "year": year,
                    "description": description,
                    "zone": precursors[shape],
                    "local_ratio": float(local_ratios[shape]),
                    "global_ratio": float(global_ratios[shape]),
                    "instance_entities": instance_entities,
                }
            )
            n_confirmed += 1

        yearly_stats.append(
            {
                "year": year,
                "precursors": len(precursors),
                "top_k": len(top_rare),
                "confirmed": n_confirmed,
            }
        )

    for t_idx, year in enumerate(years):
        vector_rows.append(
            {
                "year": year,
                "counts": {shape: yearly_counts[t_idx][shape] for shape in GRAPHLET_NAMES},
            }
        )

    detections.sort(key=lambda row: (row["year"], row["graphlet"]))
    return detections, vector_rows, yearly_stats


def save(detections: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)
        f.write("\n")


def _print_summary(
    vector_rows: list[dict],
    yearly_stats: list[dict],
    detections: list[dict],
) -> None:
    print("Graphlet vectors per year (G0..G8):")
    for row in vector_rows:
        counts = row["counts"]
        values = [counts[shape] for shape in GRAPHLET_NAMES]
        print(f"  {row['year']}: {values}")

    print("Precursors / confirmed per year:")
    for row in yearly_stats:
        print(
            f"  {row['year']}: precursors={row['precursors']} "
            f"top_k={row['top_k']} confirmed={row['confirmed']}"
        )

    by_graphlet = Counter(d["graphlet"] for d in detections)
    by_zone = Counter(d["zone"] for d in detections)
    print(f"Total detections: {len(detections)}")
    print(f"  by graphlet: {dict(sorted(by_graphlet.items()))}")
    print(f"  by zone: {dict(sorted(by_zone.items()))}")


def main() -> None:
    print("Loading snapshots from:", SNAPSHOTS_DIR)
    snapshots = load_snapshots(SNAPSHOTS_DIR)
    year_graphs = build_year_graphs(snapshots)
    print(f"Loaded {len(year_graphs)} yearly graphs")

    detections, vector_rows, yearly_stats = extract_detections(year_graphs)
    save(detections, OUTPUT_JSON)

    print("Wrote:", OUTPUT_JSON)
    _print_summary(vector_rows, yearly_stats, detections)


if __name__ == "__main__":
    print("=" * 50)
    print("  BEAM GRAPHLET WEAK-SIGNAL DETECTOR")
    print("=" * 50)
    main()
