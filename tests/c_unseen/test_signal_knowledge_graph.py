"""SignalKnowledgeGraph inheritance and JSON hydration defaults."""

from __future__ import annotations

from pathlib import Path

from itext2kg.atom.models import (
    Entity,
    KnowledgeGraph,
    Relationship,
    RelationshipProperties,
)
from itext2kg.c_unseen.models.signal_knowledge_graph import (
    SignalKnowledgeGraph,
    SignalProperties,
    SignalRelationship,
)


def _atom_kg() -> KnowledgeGraph:
    a = Entity(name="A", label="org")
    b = Entity(name="B", label="org")
    c = Entity(name="C", label="org")
    r0 = Relationship(
        name="links",
        startEntity=a,
        endEntity=b,
        properties=RelationshipProperties(domains=["x"]),
    )
    r1 = Relationship(
        name="links",
        startEntity=b,
        endEntity=c,
        properties=RelationshipProperties(domains=["y"]),
    )
    return KnowledgeGraph(entities=[a, b, c], relationships=[r0, r1])


def test_signal_flags_default_false_on_from_json(tmp_path: Path) -> None:
    kg = _atom_kg()
    path = tmp_path / "atom.json"
    kg.to_json(path)

    skg = SignalKnowledgeGraph.from_json(path)
    assert isinstance(skg, SignalKnowledgeGraph)
    assert len(skg.relationships) == 2
    for rel in skg.relationships:
        assert isinstance(rel, SignalRelationship)
        assert isinstance(rel.properties, SignalProperties)
        assert rel.properties.rare is False
        assert rel.properties.weak_signal_pred is False
        assert rel.properties.already_corroborated is False
        assert rel.properties.domains  # preserved from ATOM JSON


def test_inherits_extract_connecting_subgraph() -> None:
    a = Entity(name="A", label="org")
    b = Entity(name="B", label="org")
    c = Entity(name="C", label="org")
    r0 = SignalRelationship(
        name="ab",
        startEntity=a,
        endEntity=b,
        properties=SignalProperties(),
    )
    r1 = SignalRelationship(
        name="bc",
        startEntity=b,
        endEntity=c,
        properties=SignalProperties(),
    )
    skg = SignalKnowledgeGraph(entities=[a, b, c], relationships=[r0, r1])
    assert hasattr(skg, "extract_connecting_subgraph")
    connected = skg.extract_connecting_subgraph([0, 1])
    assert 0 in connected and 1 in connected


def test_signal_properties_inherits_new_atom_fields() -> None:
    assert "domains" in SignalProperties.model_fields
    assert "atomic_facts" in SignalProperties.model_fields
    assert "rare" in SignalProperties.model_fields
    props = SignalProperties(domains=["finance"], rare=True)
    assert props.domains == ["finance"]
    assert props.rare is True
