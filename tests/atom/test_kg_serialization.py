"""JSON + NPZ round-trip tests for ATOM KnowledgeGraph persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from itext2kg.atom.models import (
    Entity,
    EntityProperties,
    KnowledgeGraph,
    Relationship,
    RelationshipProperties,
)


def _make_kg() -> KnowledgeGraph:
    a = Entity(
        name="Alpha",
        label="org",
        properties=EntityProperties(embeddings=np.arange(4, dtype=np.float64)),
    )
    b = Entity(
        name="Beta",
        label="org",
        properties=EntityProperties(embeddings=np.arange(4, 8, dtype=np.float64)),
    )
    rel = Relationship(
        name="partners_with",
        startEntity=a,
        endEntity=b,
        properties=RelationshipProperties(
            embeddings=np.arange(8, 12, dtype=np.float64),
            atomic_facts=["Alpha partners with Beta in 2016."],
            domains=["business"],
            t_obs=[1451606400.0],
            t_start=[1451606400.0],
            t_end=[1483142400.0],
        ),
    )
    return KnowledgeGraph(entities=[a, b], relationships=[rel])


def test_json_round_trip_preserves_structure(tmp_path: Path) -> None:
    kg = _make_kg()
    json_path = tmp_path / "kg.json"
    kg.to_json(json_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert "embeddings" not in payload["entities"][0]
    assert "embeddings" not in payload["relationships"][0]["properties"]

    loaded = KnowledgeGraph.from_json(json_path)
    assert len(loaded.entities) == 2
    assert len(loaded.relationships) == 1
    rel = loaded.relationships[0]
    assert rel.name == "partners_with"
    assert rel.startEntity.name == "Alpha"
    assert rel.endEntity.name == "Beta"
    assert rel.properties.atomic_facts == ["Alpha partners with Beta in 2016."]
    assert rel.properties.domains == ["business"]
    assert rel.properties.t_obs == [1451606400.0]
    assert rel.properties.t_start == [1451606400.0]
    assert rel.properties.t_end == [1483142400.0]
    assert loaded.entities[0].properties.embeddings is None
    assert rel.properties.embeddings is None


def test_npz_round_trip_preserves_embeddings(tmp_path: Path) -> None:
    kg = _make_kg()
    json_path = tmp_path / "kg.json"
    npz_path = tmp_path / "kg.npz"
    kg.to_json(json_path, embeddings_path=npz_path)

    loaded = KnowledgeGraph.from_json(json_path, embeddings_path=npz_path)
    assert loaded.entities[0].properties.embeddings is not None
    assert loaded.entities[1].properties.embeddings is not None
    assert loaded.relationships[0].properties.embeddings is not None
    np.testing.assert_allclose(
        loaded.entities[0].properties.embeddings,
        np.arange(4, dtype=np.float32),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        loaded.relationships[0].properties.embeddings,
        np.arange(8, 12, dtype=np.float32),
        rtol=0,
        atol=0,
    )
    assert loaded.entities[0].properties.embeddings.dtype == np.float32


def test_json_without_npz_leaves_embeddings_none(tmp_path: Path) -> None:
    kg = _make_kg()
    json_path = tmp_path / "kg.json"
    npz_path = tmp_path / "kg.npz"
    kg.to_json(json_path, embeddings_path=npz_path)

    loaded = KnowledgeGraph.from_json(json_path)
    assert all(e.properties.embeddings is None for e in loaded.entities)
    assert all(r.properties.embeddings is None for r in loaded.relationships)
