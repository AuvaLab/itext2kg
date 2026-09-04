"""Domain propagation through ATOM relationships and C-Unseen detector formatting."""

from __future__ import annotations

from itext2kg.atom.models import (
    Entity,
    KnowledgeGraph,
    Relationship,
    RelationshipProperties,
)
from itext2kg.c_unseen.rare_elements_detector.detector import RareElementsDetector


def _rel(*, domains: list[str] | None = None) -> Relationship:
    a = Entity(name="StartCo", label="org")
    b = Entity(name="EndCo", label="org")
    return Relationship(
        name="funds",
        startEntity=a,
        endEntity=b,
        properties=RelationshipProperties(
            domains=list(domains or []),
            t_start=[1483228800.0],
            t_end=[1496275200.0],
        ),
    )


def test_default_domains_empty_and_detector_shows_n_a() -> None:
    rel = _rel()
    assert rel.properties.domains == []
    line = RareElementsDetector._format_indexed_triple(0, rel)
    assert "domain=n/a" in line


def test_add_domains_to_relationships_propagates() -> None:
    rel = _rel()
    kg = KnowledgeGraph(entities=[rel.startEntity, rel.endEntity], relationships=[rel])
    kg.add_domains_to_relationships(["research"])
    assert kg.relationships[0].properties.domains == ["research"]
    line = RareElementsDetector._format_indexed_triple(3, kg.relationships[0])
    assert line.startswith("[3] ")
    assert "domain=research" in line


def test_combine_domains_appends() -> None:
    rel = _rel(domains=["a"])
    rel.combine_domains(["a", "b"])
    assert rel.properties.domains == ["a", "a", "b"]
