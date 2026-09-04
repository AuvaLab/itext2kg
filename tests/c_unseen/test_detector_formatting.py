"""Detector triple / quintuple formatting must match the prompt template shape."""

from __future__ import annotations

from itext2kg.atom.models import Entity, Relationship, RelationshipProperties
from itext2kg.c_unseen.rare_elements_detector.detector import (
    RareElementsDetector,
    format_quintuple,
)


def test_format_quintuple_shape() -> None:
    rel = Relationship(
        name="announces",
        startEntity=Entity(name="OpenAI", label="company"),
        endEntity=Entity(name="GPT-3", label="product"),
        properties=RelationshipProperties(
            t_start=[1590624000.0],
            t_end=[1590624000.0],
        ),
    )
    assert format_quintuple(rel) == (
        "(OpenAI: company) --> announces (2020-05-28, 2020-05-28) --> (GPT-3: product)"
    )


def test_format_indexed_triple_with_domain() -> None:
    rel = Relationship(
        name="raises",
        startEntity=Entity(name="OpenAI", label="company"),
        endEntity=Entity(name="Microsoft", label="company"),
        properties=RelationshipProperties(
            domains=["funding", "funding", "partnership"],
            t_start=[1563753600.0],
            t_end=[],
        ),
    )
    line = RareElementsDetector._format_indexed_triple(2, rel)
    assert line == (
        "[2] (OpenAI: company) --> raises (2019-07-22, ) --> "
        "(Microsoft: company) | domain=funding"
    )


def test_format_indexed_triple_empty_domain_is_n_a() -> None:
    rel = Relationship(
        name="mentions",
        startEntity=Entity(name="A", label=""),
        endEntity=Entity(name="B", label=""),
        properties=RelationshipProperties(),
    )
    line = RareElementsDetector._format_indexed_triple(0, rel)
    assert line == "[0] (A: entity) --> mentions (, ) --> (B: entity) | domain=n/a"
