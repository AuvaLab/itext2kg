#!/usr/bin/env python3
"""
Build a static ATOM KG from a single PDF (one observation time).

1. Extract text with PyMuPDF (sort=True) and apply light cleaning.
2. If the document has <= WHOLE_DOC_TOKEN_LIMIT tokens, pass it whole;
   otherwise split with SemanticChunker.
3. Extract atomic facts per unit; cache them for resume.
4. Build one KG at OBS_TIMESTAMP and save JSON + NPZ.
5. Optionally push to Neo4j.

Usage:
    pyenv activate venv-itext2kg-update
    python useful_scripts/build_atom_kg_from_pdf.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pymupdf
import tiktoken
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from itext2kg.atom.atom import Atom
from itext2kg.atom.models import KnowledgeGraph
from itext2kg.atom.models.schemas import AtomicFact
from itext2kg.graph_integration import Neo4jStorage
from itext2kg.llm_output_parsing import LangchainOutputParser
from itext2kg.logging_config import setup_logging

# ==========================
# User-configurable globals
# ==========================
INPUT_PDF_PATH = PROJECT_ROOT / "datasets" / "itext2kg" / "cvs" / "CV_Emily_Davis.pdf"
OUTPUT_DIR = INPUT_PDF_PATH.parent / "atom_kg"

OBS_TIMESTAMP = "2026-09-04"  # single observation time; a CV carries no date
WHOLE_DOC_TOKEN_LIMIT = 1500  # <= this, skip chunking entirely
TOKEN_ENCODING = "cl100k_base"  # matches LangchainOutputParser.count_tokens

BREAKPOINT_THRESHOLD_TYPE = "percentile"
BREAKPOINT_THRESHOLD_AMOUNT = 95.0
SENTENCE_SPLIT_REGEX = r"(?<=[.?!])\s+|\n{2,}"
MIN_CHUNK_SIZE = 200  # chars, suppresses one-line chunks

OUTPUT_KG_JSON = OUTPUT_DIR / f"{INPUT_PDF_PATH.stem}_kg.json"
OUTPUT_KG_NPZ = OUTPUT_DIR / f"{INPUT_PDF_PATH.stem}_kg.npz"
TEXT_CACHE_PATH = OUTPUT_DIR / f"{INPUT_PDF_PATH.stem}_text.txt"
FACTS_CACHE_PATH = OUTPUT_DIR / f"{INPUT_PDF_PATH.stem}_facts.json"
UNITS_CACHE_PATH = OUTPUT_DIR / f"{INPUT_PDF_PATH.stem}_units.json"
LOG_FILE = OUTPUT_DIR / "build_atom_kg_from_pdf.log"

SEND_TO_NEO4J = False
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "oNews")

OPENAI_MODEL_NAME = "gpt-4.1-2025-04-14"
OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-large"

ENT_THRESHOLD = 0.8
REL_THRESHOLD = 0.7
ENTITY_NAME_WEIGHT = 0.8
ENTITY_LABEL_WEIGHT = 0.2
MAX_WORKERS = 8

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def configure_logging() -> logging.Logger:
    """Dual logging: console + file (script logger and itext2kg)."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level="INFO",
        format_string=LOG_FORMAT,
        log_file=str(LOG_FILE),
        console_output=True,
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


logger = configure_logging()


def log_phase(title: str) -> None:
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


def resolve_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        from api_keys import openai_api_key

        if openai_api_key:
            return openai_api_key
    except ImportError:
        pass
    raise RuntimeError(
        "OpenAI API key not set. Export OPENAI_API_KEY or add openai_api_key to api_keys.py."
    )


def resolve_neo4j_password() -> str:
    if NEO4J_PASSWORD:
        return NEO4J_PASSWORD
    try:
        from api_keys import neo4_password

        return neo4_password
    except ImportError:
        pass
    raise RuntimeError(
        "Neo4j password not set. Export NEO4J_PASSWORD or add neo4_password to api_keys.py."
    )


def count_tokens(text: str, encoding_name: str = TOKEN_ENCODING) -> int:
    if not text:
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def atomic_write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Phase 1 — PDF extraction + cleaning
# ---------------------------------------------------------------------------


def extract_pdf_text(path: Path) -> str:
    """Extract page text with PyMuPDF, ordering blocks by reading position."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input PDF: {path}")
    doc = pymupdf.open(path)
    try:
        pages = [
            page.get_text("text", sort=True).strip()
            for page in doc
            if page.get_text("text", sort=True).strip()
        ]
    finally:
        doc.close()
    return "\n\n".join(pages)


def _drop_repeated_headers_footers(text: str, min_pages: int = 2) -> str:
    """Drop lines that appear on most pages (running headers / footers)."""
    pages = [p for p in text.split("\n\n") if p.strip()]
    if len(pages) < min_pages:
        return text

    line_counts: Counter[str] = Counter()
    for page in pages:
        unique_lines = {ln.strip() for ln in page.splitlines() if ln.strip()}
        line_counts.update(unique_lines)

    threshold = max(2, (len(pages) + 1) // 2)
    repeated = {ln for ln, n in line_counts.items() if n >= threshold and len(ln) < 120}
    if not repeated:
        return text

    cleaned_pages: list[str] = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if ln.strip() not in repeated]
        cleaned_pages.append("\n".join(kept).strip())
    return "\n\n".join(p for p in cleaned_pages if p)


def clean_pdf_text(text: str) -> str:
    """Light generic cleaning for PDF-extracted text."""
    if not text:
        return ""
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _drop_repeated_headers_footers(text)
    lines = []
    for ln in text.splitlines():
        if _PAGE_NUMBER_RE.match(ln):
            continue
        lines.append(ln)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def phase_extract_text() -> str:
    log_phase("PHASE 1/4 — PDF extraction + cleaning")
    t0 = time.time()
    raw = extract_pdf_text(INPUT_PDF_PATH)
    cleaned = clean_pdf_text(raw)
    atomic_write_text(cleaned, TEXT_CACHE_PATH)
    n_tokens = count_tokens(cleaned)
    logger.info(
        "Extracted %d chars / %d tokens → %s (%.1fs)",
        len(cleaned),
        n_tokens,
        TEXT_CACHE_PATH,
        time.time() - t0,
    )
    return cleaned


# ---------------------------------------------------------------------------
# Phase 2 — token gate / semantic chunking
# ---------------------------------------------------------------------------


def phase_chunk(text: str, embeddings) -> list[str]:
    log_phase("PHASE 2/4 — Token gate / semantic chunking")
    n_tokens = count_tokens(text)
    logger.info(
        "Document tokens=%d | WHOLE_DOC_TOKEN_LIMIT=%d",
        n_tokens,
        WHOLE_DOC_TOKEN_LIMIT,
    )

    if n_tokens <= WHOLE_DOC_TOKEN_LIMIT:
        logger.info(
            "Whole-document path — skipping SemanticChunker (tokens %d <= %d)",
            n_tokens,
            WHOLE_DOC_TOKEN_LIMIT,
        )
        units = [text]
    else:
        logger.info(
            "SemanticChunker path — breakpoint=%s/%s min_chunk_size=%d",
            BREAKPOINT_THRESHOLD_TYPE,
            BREAKPOINT_THRESHOLD_AMOUNT,
            MIN_CHUNK_SIZE,
        )
        chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=BREAKPOINT_THRESHOLD_TYPE,
            breakpoint_threshold_amount=BREAKPOINT_THRESHOLD_AMOUNT,
            sentence_split_regex=SENTENCE_SPLIT_REGEX,
            min_chunk_size=MIN_CHUNK_SIZE,
        )
        units = [u.strip() for u in chunker.split_text(text) if u and u.strip()]
        if not units:
            logger.warning("SemanticChunker returned no units; falling back to whole document")
            units = [text]

    for i, unit in enumerate(units):
        logger.info("unit %d/%d — %d tokens", i, len(units) - 1, count_tokens(unit))

    atomic_write_json({"obs_timestamp": OBS_TIMESTAMP, "units": units}, UNITS_CACHE_PATH)
    logger.info("Cached %d units → %s", len(units), UNITS_CACHE_PATH)
    return units


# ---------------------------------------------------------------------------
# Phase 3 — atomic facts (resumable)
# ---------------------------------------------------------------------------


async def phase_extract_facts(
    parser: LangchainOutputParser, units: list[str]
) -> list[str]:
    log_phase("PHASE 3/4 — Atomic facts extraction")

    if FACTS_CACHE_PATH.exists():
        with open(FACTS_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        facts = list(cached.get("atomic_facts") or [])
        logger.info(
            "Facts cache hit — loaded %d facts from %s; skipping LLM extraction",
            len(facts),
            FACTS_CACHE_PATH,
        )
        return facts

    if not units:
        logger.warning("No units to extract from; writing empty facts cache")
        atomic_write_json(
            {"obs_timestamp": OBS_TIMESTAMP, "atomic_facts": [], "per_unit": []},
            FACTS_CACHE_PATH,
        )
        return []

    logger.info("Calling LLM for %d units...", len(units))
    t0 = time.time()
    results = await parser.extract_information_as_json_for_context(
        AtomicFact, units, return_exceptions=True
    )

    pooled: list[str] = []
    per_unit: list[dict] = []
    n_ok = 0
    n_fail = 0
    for i, (unit, out) in enumerate(zip(units, results)):
        if isinstance(out, BaseException):
            logger.warning("unit %d/%d FAILED: %s", i, len(units) - 1, out)
            n_fail += 1
            per_unit.append({"index": i, "error": str(out), "atomic_facts": []})
            continue
        facts = list(out.atomic_fact) if out else []
        pooled.extend(facts)
        per_unit.append({"index": i, "atomic_facts": facts})
        n_ok += 1
        logger.info(
            "unit %d/%d OK — %d atomic facts (elapsed %.1fs)",
            i,
            len(units) - 1,
            len(facts),
            time.time() - t0,
        )

    atomic_write_json(
        {
            "obs_timestamp": OBS_TIMESTAMP,
            "atomic_facts": pooled,
            "per_unit": per_unit,
        },
        FACTS_CACHE_PATH,
    )
    logger.info(
        "Phase 3 done in %.1fs — %d successes, %d failures, %d pooled facts → %s",
        time.time() - t0,
        n_ok,
        n_fail,
        len(pooled),
        FACTS_CACHE_PATH,
    )
    if n_fail and not pooled:
        raise RuntimeError(
            f"All {n_fail} unit extraction(s) failed; nothing to build a KG from."
        )
    return pooled


# ---------------------------------------------------------------------------
# Phase 4 — single static KG
# ---------------------------------------------------------------------------


def save_kg(kg: KnowledgeGraph) -> None:
    """Atomic JSON + NPZ write (temp NPZ must end in .npz)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_json = OUTPUT_DIR / f".{OUTPUT_KG_JSON.name}.tmp.json"
    tmp_npz = OUTPUT_DIR / f".{OUTPUT_KG_NPZ.stem}.tmp.npz"
    kg.to_json(tmp_json, embeddings_path=tmp_npz)
    os.replace(tmp_json, OUTPUT_KG_JSON)
    os.replace(tmp_npz, OUTPUT_KG_NPZ)


async def phase_build_kg(atom: Atom, facts: list[str]) -> KnowledgeGraph:
    log_phase("PHASE 4/4 — Build static KG")
    t0 = time.time()
    if not facts:
        logger.warning("Empty atomic_facts → empty KG")
        kg = KnowledgeGraph()
    else:
        logger.info(
            "Building KG from %d facts at obs_timestamp=%s ...",
            len(facts),
            OBS_TIMESTAMP,
        )
        kg = await atom.build_graph(
            atomic_facts=facts,
            obs_timestamp=OBS_TIMESTAMP,
            ent_threshold=ENT_THRESHOLD,
            rel_threshold=REL_THRESHOLD,
            entity_name_weight=ENTITY_NAME_WEIGHT,
            entity_label_weight=ENTITY_LABEL_WEIGHT,
            max_workers=MAX_WORKERS,
        )
    save_kg(kg)
    logger.info(
        "Phase 4 done in %.1fs — saved %s (+ %s) (%d entities, %d relationships)",
        time.time() - t0,
        OUTPUT_KG_JSON,
        OUTPUT_KG_NPZ.name,
        len(kg.entities),
        len(kg.relationships),
    )
    return kg


def push_to_neo4j(kg: KnowledgeGraph) -> None:
    log_phase("OPTIONAL — Push to Neo4j")
    logger.info(
        "Connecting to Neo4j uri=%s user=%s database=%s",
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_DATABASE,
    )
    t0 = time.time()
    storage = Neo4jStorage(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=resolve_neo4j_password(),
        database=NEO4J_DATABASE,
    )
    logger.info(
        "Pushing KG (%d entities, %d relationships)...",
        len(kg.entities),
        len(kg.relationships),
    )
    storage.visualize_graph(kg)
    logger.info("Neo4j load complete in %.1fs.", time.time() - t0)


async def main() -> None:
    start = time.time()
    log_phase("BUILD ATOM KG FROM PDF — start")
    logger.info("Log file: %s", LOG_FILE.resolve())
    logger.info("Input PDF: %s", INPUT_PDF_PATH)
    logger.info("Output dir: %s", OUTPUT_DIR)
    logger.info("Output KG: %s (+ %s)", OUTPUT_KG_JSON, OUTPUT_KG_NPZ.name)
    logger.info(
        "Config: OBS_TIMESTAMP=%s | WHOLE_DOC_TOKEN_LIMIT=%d | SEND_TO_NEO4J=%s | model=%s",
        OBS_TIMESTAMP,
        WHOLE_DOC_TOKEN_LIMIT,
        SEND_TO_NEO4J,
        OPENAI_MODEL_NAME,
    )

    logger.info("Bootstrapping OpenAI LLM + embeddings...")
    api_key = resolve_openai_api_key()
    openai_llm = ChatOpenAI(
        api_key=api_key,
        model=OPENAI_MODEL_NAME,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    openai_emb = OpenAIEmbeddings(api_key=api_key, model=OPENAI_EMBEDDINGS_MODEL)
    parser = LangchainOutputParser(llm_model=openai_llm, embeddings_model=openai_emb)
    atom = Atom(llm_model=openai_llm, embeddings_model=openai_emb)
    logger.info("Models ready.")

    text = phase_extract_text()
    units = phase_chunk(text, openai_emb)
    facts = await phase_extract_facts(parser, units)
    final_kg = await phase_build_kg(atom, facts)

    if SEND_TO_NEO4J:
        push_to_neo4j(final_kg)
    else:
        log_phase("OPTIONAL — Neo4j skipped")
        logger.info("SEND_TO_NEO4J=False; skipped Neo4j upload.")

    elapsed = time.time() - start
    log_phase("DONE")
    logger.info(
        "Finished in %.1fs | KG: %s (+ %s) | facts: %s | log: %s",
        elapsed,
        OUTPUT_KG_JSON,
        OUTPUT_KG_NPZ.name,
        FACTS_CACHE_PATH,
        LOG_FILE,
    )


if __name__ == "__main__":
    asyncio.run(main())
