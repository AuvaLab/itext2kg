#!/usr/bin/env python3
"""
Ablation study for C-Unseen on the Wiki-OpenAI weak-signal benchmark.

Holds the LLM and the output schema fixed. Varies (a) the prompt and (b) whether
the model has memory of earlier years:

    arm  ablation_flat_local   DTKG quintuples, current year only, flat prompt
    arm  ablation_cot_local    DTKG quintuples, current year only, CoT prompt
    arm  ablation_flat_facts   DTKG quintuples, years <= n as context, flat prompt
    arm  ablation_cot_facts    DTKG quintuples, years <= n as context, CoT prompt
    arm  c_unseen              DTKG + CoT  (run RUN_TIMES times, same as the LLM arms)

MEMORY vs LOCAL
Local arms see only year n. Memory arms see years < n as unflaggable context
plus year n as the flaggable list. The local -> memory delta is the value of
memory. C-Unseen is the memory arm that also reasons over the DTKG, with
C-Unseen's own rare/bridging pipeline rather than a flat quintuple dump.

Local CoT is within-snapshot reasoning only (baseline + 7 deviation types +
grouping). It does not mention earlier years. Reusing the memory CoT with an
empty past block would ask the model to develop past tensions it cannot see.

TEMPORAL PROTOCOL
All four LLM arms are causal: one call per target year, never the future.
The first year in range is never a target, because C-Unseen has no bridging
trace for the first snapshot. Same target years across arms.

Setting CALL_GRANULARITY = "all_years" instead puts the whole corpus in one
call. That is a hindsight oracle, not a fair comparison against C-Unseen.

INPUT
LLM arms see the full yearly DTKG as numbered quintuples:
    (subject name: type) --> predicate (t_start, t_end) --> (object name: type)
t_start / t_end may be empty. Roles and atomic-fact text are not shown.

LEAKAGE
assert_no_leakage() hard-fails if role / precursor / rationale tokens ever
reach the prompt.

REPLICATES
Each LLM arm, including C-Unseen, is run RUN_TIMES times (default 5). Caches
are per (arm, run). The first existing cache (legacy, no run suffix) is reused
as run 0 so previous calls are not wasted. C-Unseen run 0 reuses
detections_c_unseen.json; later runs write isolated traces under
evaluation/c_unseen/reports/ablation/c_unseen_runs/run{N}. Reported tables are
mean ± sample standard deviation across replicates.

Usage:
    pyenv activate venv-itext2kg-update
    python evaluation/c_unseen/run/ablation_c_unseen.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

for _p in Path(__file__).resolve().parents:
    if (_p / ".git").exists():
        project_root = _p
        break
else:
    raise RuntimeError("Could not locate repository root")
sys.path.insert(0, str(project_root))

from api_keys import openai_api_key  # noqa: E402
from evaluation.c_unseen.paths import (  # noqa: E402
    BENCHMARK,
    DETECTORS,
    OUTPUTS,
    REPORTS,
    SCORING,
    SNAPSHOTS,
)

# ==========================
# User-configurable globals
# ==========================
BENCHMARK_JSON: Path = BENCHMARK
SNAPSHOTS_DIR: Path = SNAPSHOTS
OUTPUT_DIR: Path = OUTPUTS
ABLATION_DIR: Path = OUTPUT_DIR / "ablation"
LLM_ABLATION_DIR: Path = OUTPUT_DIR / "ablation_dtkg"
TRACES_DIR: Path = REPORTS / "ablation"

# Arm 3 is produced by c_unseen_runner.py (reused as-is, not recomputed).
C_UNSEEN_DETECTIONS: Path = OUTPUT_DIR / "detections_c_unseen.json"

YEAR_START: int = 2015
YEAR_END: int = 2025

# Must match the model that produced detections_c_unseen.json, otherwise the
# ablation confounds prompt/representation with model capability.
OPENAI_MODEL_NAME: str = "gpt-5.4-mini-2026-03-17"
OPENAI_EMBEDDINGS_MODEL: str = "text-embedding-3-large"
TEMPERATURE: int = 0
MAX_TOKENS = None
TIMEOUT = None
MAX_RETRIES: int = 2

# "expanding_window": causal, one call per target year, past years as context.
#                     The only setting comparable to C-Unseen.
# "all_years":        whole corpus in one call. Hindsight oracle / upper bound.
CALL_GRANULARITY: str = "expanding_window"

# Arm 3 has no bridging trace for the first snapshot, so the first year in range
# is context-only and never a detection target.
SKIP_FIRST_YEAR_AS_TARGET: bool = True

# "max": detection year is the latest year among the facts in the group (earliest
#        point at which all evidence exists). Only bites in "all_years" mode; in
#        expanding-window mode every flagged fact already belongs to the target year.
DETECTION_YEAR_RULE: str = "max"

# Mirrors run_c_unseen_whole_snapshot._weak_buckets_from_trace: when the model
# returns explanations they define the detections, and weak_signal_indices is
# used only as a fallback when explanations is empty. Setting this True instead
# emits an extra singleton detection per flagged-but-unexplained index, which
# inflates the detection count and is not what arm 3 does.
EMIT_UNEXPLAINED_INDICES: bool = False

FORCE_RERUN: bool = False

# Independent LLM replicates. Each arm, including C-Unseen, is called
# RUN_TIMES times; reported metrics are mean ± sample std. Run 0 reuses
# existing caches when present (legacy fact-arm caches and detections_c_unseen.json).
RUN_TIMES: int = 5

# Bumped whenever the prompts change, so stale caches are never silently reused.
PROMPT_VERSION: str = "v4_dtkg_grounded"

_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")

# Substrings that must never appear in the fact blocks sent to the LLM.
FORBIDDEN_IN_FACT_BLOCK: tuple[str, ...] = (
    "precursor_of",
    "off_narrative_rationale",
    "lead_time_days",
    "lead_tier",
    "provenance",
    "event_date",
    "strong_signal",
    "weak_signal",
    "corroboration",
    "SS_",
)


# ==========================
# Prompts
# ==========================
FLAT_LOCAL_SYSTEM = """You are an analyst. Detect weak signals in the DTKG quintuples below."""

FLAT_LOCAL_USER = """Scope: {scope_label}

Current-period DTKG quintuples:
{current_facts}

Each line is: (subject name: type) --> predicate (t_start, t_end) --> (object name: type). t_start and t_end may be empty.

Return structured output with:
- weak_signal_indices: flat deduplicated list of quintuple indices you flag as weak signals
- explanations: one item per weak signal; index is the list of quintuple indices forming it, plus theme and explanation"""

FLAT_LOCAL_PROMPT = ChatPromptTemplate.from_messages(
    [("system", FLAT_LOCAL_SYSTEM), ("human", FLAT_LOCAL_USER)]
)

FLAT_MEMORY_SYSTEM = """You are an analyst. Detect weak signals in the current-period DTKG quintuples below."""

FLAT_MEMORY_USER = """Scope: {scope_label}

DTKG quintuples from earlier periods (context only, never flag these):
{past_facts}

Current-period DTKG quintuples (flag only these):
{current_facts}

Each line is: (subject name: type) --> predicate (t_start, t_end) --> (object name: type). t_start and t_end may be empty.

Return structured output with:
- weak_signal_indices: flat deduplicated list of current-period quintuple indices you flag as weak signals
- explanations: one item per weak signal; index is the list of current-period quintuple indices forming it, plus theme and explanation"""

FLAT_MEMORY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", FLAT_MEMORY_SYSTEM), ("human", FLAT_MEMORY_USER)]
)

COT_LOCAL_SYSTEM = """You are an analyst specializing in early-warning signals extracted from a single time window of a dynamic temporal knowledge graph.

---

## What you receive

A numbered list of DTKG quintuples from the CURRENT period's full graph only:

    [index] (subject name: type) --> predicate (t_start, t_end) --> (object name: type)

t_start and t_end may be empty. You do not receive earlier periods. You do not know what happens after this period. Never assume an outcome the provided quintuples do not already state.

## Grounding (hard constraint)

Reason only from the numbered quintuples in this prompt. Treat the central entity as unknown besides those quintuples. Do not use prior knowledge of this company, its later outcomes, or famous events. If justifying a flag requires an outcome, motive, or later history that is not stated in the provided quintuples, do not flag it. Reading several provided quintuples together is allowed; filling gaps from what you know is not.

---

## Your core task

Identify which facts, individually or as a group, are WEAK SIGNALS.

A weak signal is a fact, or a group of facts that must be read together, that is off-narrative relative to this period's dominant story: semantically meaningful, marginal in this snapshot, and worth tracking because it hints at a process, tension, or trajectory the dominant narrative does not account for.

---

## Step 1 - Establish the baseline narrative

Read all facts and answer in plain language:
- What is the central entity's dominant situation in this period?
- Who are the main actors, and what are they doing?
- What is the prevailing story these facts collectively tell?

The baseline is the majority story. A fact is only off-narrative relative to that baseline.

---

## Step 2 - Find the off-narrative facts

Scan every fact for the following deviation types. A deviation is real only if it is grounded in the real-world content of the fact, not in its wording or its position in the list.

### Type 1 - Quantitative asymmetry
Two or more facts together reveal a gap between a stated figure and a realized figure: a pledge versus what was collected, a claimed capacity versus what was delivered, an announced target versus an actual outcome. The gap is the signal, not either number alone.

### Type 2 - Covert trajectory
A fact reveals that a consequential decision or shift is already underway in private, while the baseline presents the situation as stable or different.

### Type 3 - Actor role conflict
A key actor appears in two or more roles that are structurally or logically in tension: co-founder and funding withholder, declared partner and active adversary, safety guardian and commercial accelerant.

### Type 4 - Policy-behavior gap
A fact describes behavior by the entity or a key actor that directly contradicts a stated rule, declared mission, or explicit commitment visible elsewhere in this period.

### Type 5 - Governance or structural hollowing
A fact, or a sequence of facts read together, describes a change in board composition, ownership structure, decision authority, or control mechanism that is framed as routine but cumulatively removes a safeguard or concentrates power.

### Type 6 - Constraint hidden inside a routine fact
A fact that looks like a standard transaction, appointment, or announcement carries an embedded clause, condition, or asymmetry imposing a hard constraint on the entity's future options: a deadline, a penalty, a dependency, a legal obligation.

### Type 7 - Domain intrusion
A fact belongs to a subject area strongly at odds with the dominant subject areas of this period. A legal, regulatory, military, or controversy fact surfacing in a period otherwise dominated by product, finance, or personnel facts, or the reverse, implies an exposure the main narrative does not address.

---

## Step 3 - Group facts when the signal requires it

Some signals become visible only when two or more facts are read together. A single departure is noise; three sequential departures that reduce a board to a removal-capable minimum is a signal. A pledge alone is unremarkable; a pledge paired with a collection figure that reveals a shortfall is a signal.

When the deviation only exists at the group level:
- Add ONE explanations entry for the whole group.
- Set index to the list of all fact indices in that group.
- Write a single shared theme and explanation.
- Also include every index in weak_signal_indices.

For a single-fact weak signal, use a one-element list.

---

## Strict ignore list

Do NOT flag:
- Facts that restate the same real-world content as another fact with different wording.
- Facts whose only anomaly is unusual phrasing rather than unusual content.
- Facts that are simply less common topically but carry no tension with this period's baseline.
- Connections that require inferring facts, causal mechanisms, or relationships not present in the list.
- Flags that depend on prior knowledge of the company, or of later events, that are not stated in the provided quintuples.
- Groups formed only because the facts share an actor or a topic. Grouping is valid only when the facts jointly constitute a step that none captures alone.

In general: if the only reason to flag a fact is how it is written rather than what it says about the world, do not flag it.

---

## Calibration rules

- Return indices into the provided list only.
- Prefer precision over recall. A vague or marginal deviation is not a weak signal.
- The baseline is the reference. A fact is not off-narrative in isolation.
- No speculation, and no knowledge of later or earlier periods.
- Reason only from the numbered quintuples in this prompt. Treat the central entity as unknown besides those quintuples.
- Do not group facts merely to manufacture a signal.
- If no fact qualifies, return empty lists.

---

## Structured output (required)

Return a single JSON object with exactly these fields:

### weak_signal_indices (list of integers)
Flat, deduplicated list of every flagged fact index. Must equal the union of all index lists in explanations.

### explanations (list of objects)
One object per weak signal (single fact or group). Each object has:

- index (list of integers): one or more fact indices forming this weak signal. Use a list even for a single fact.
- theme (string): the underlying tension this signal encodes.
- explanation (string): the deviation type, the real-world content, and the tension versus this period's baseline. For groups, explain why the signal exists only when the facts are read together."""

COT_LOCAL_USER = """Scope: {scope_label}

Current-period DTKG quintuples:
{current_facts}

Return structured output with:
- weak_signal_indices: flat deduplicated list of flagged quintuple indices
- explanations: one item per weak signal; each index is a list of quintuple indices sharing the same theme and explanation"""

COT_LOCAL_PROMPT = ChatPromptTemplate.from_messages(
    [("system", COT_LOCAL_SYSTEM), ("human", COT_LOCAL_USER)]
)

COT_SYSTEM = """You are a weak-signal analyst tracking a central entity over time from a dynamic temporal knowledge graph.

---

## What you receive

1. EARLIER-PERIOD QUINTUPLES - DTKG quintuples from before the current period, grouped by period. These are context. You never flag them.
2. CURRENT-PERIOD QUINTUPLES - DTKG quintuples from the current period, each with an index. These are the only items you may flag.

Each quintuple is one relationship:
    (subject name: type) --> predicate (t_start, t_end) --> (object name: type)
t_start and t_end may be empty.

You do not know what happens after the current period. Never assume an outcome that the provided quintuples do not already state.

## Grounding (hard constraint)

Reason only from the numbered quintuples in this prompt. Treat the central entity as unknown besides those quintuples. Do not use prior knowledge of this company, its later outcomes, or famous events. If justifying a weak signal requires an outcome, motive, or later history that is not stated in the provided quintuples, do not flag it. Reading several provided quintuples together is allowed; filling gaps from what you know is not.

---

## Your core task

Identify which current-period facts, individually or as a group, are WEAK SIGNALS.

A weak signal is a current-period fact, or a group of current-period facts that must be read together, that is off-narrative relative to the current period's dominant story AND develops a tension that an earlier period already hinted at. It makes that earlier marginal fact look less like noise and more like the beginning of a traceable, consequential pattern.

A fact that merely shares a topic, a domain, or an actor name with an earlier fact is NOT a weak signal.

---

## Step 1 - Establish the current period's baseline narrative

Read all current-period facts and answer in plain language:
- What is the central entity's dominant situation in this period?
- Who are the main actors, and what are they doing?
- What is the prevailing story these facts collectively tell?

The baseline is the majority story of the current period. A fact is only off-narrative relative to that baseline.

---

## Step 2 - Find the off-narrative current-period facts

Scan every current-period fact for the following deviation types. A deviation is real only if it is grounded in the real-world content of the fact, not in its wording or its position in the list.

### Type 1 - Quantitative asymmetry
Two or more facts together reveal a gap between a stated figure and a realized figure: a pledge versus what was collected, a claimed capacity versus what was delivered, an announced target versus an actual outcome. The gap is the signal, not either number alone.

### Type 2 - Covert trajectory
A fact reveals that a consequential decision or shift is already underway in private, while the baseline presents the situation as stable or different.

### Type 3 - Actor role conflict
A key actor appears in two or more roles that are structurally or logically in tension: co-founder and funding withholder, declared partner and active adversary, safety guardian and commercial accelerant.

### Type 4 - Policy-behavior gap
A fact describes behavior by the entity or a key actor that directly contradicts a stated rule, declared mission, or explicit commitment visible elsewhere in the corpus.

### Type 5 - Governance or structural hollowing
A fact, or a sequence of facts read together, describes a change in board composition, ownership structure, decision authority, or control mechanism that is framed as routine but cumulatively removes a safeguard or concentrates power.

### Type 6 - Constraint hidden inside a routine fact
A fact that looks like a standard transaction, appointment, or announcement carries an embedded clause, condition, or asymmetry imposing a hard constraint on the entity's future options: a deadline, a penalty, a dependency, a legal obligation.

### Type 7 - Domain intrusion
A fact belongs to a subject area strongly at odds with the dominant subject areas of the current period. A legal, regulatory, military, or controversy fact surfacing in a period otherwise dominated by product, finance, or personnel facts, or the reverse, implies an exposure the main narrative does not address.

---

## Step 3 - Extract the underlying tension of each earlier period

Before judging the current facts, read the earlier-period facts and, for each earlier period, answer:
- What was the dominant narrative of that period?
- What tension, gap, or hidden process sat beneath it?
- In one sentence, what was that period hinting at?

Name each tension explicitly. These are your reference points.

---

## Step 4 - Test each current-period fact against each earlier tension

For every current-period fact, and every meaningful combination of them, ask of each earlier tension: does this fact, or this group read together, advance, deepen, or make more explicit the tension that earlier period encoded?

Look for these modes of development:

COVERT to OVERT - a decision or behavior that was private, internal, or denied earlier is now publicly confirmed, institutionalized, or officially acknowledged.

ISOLATED INSTANCE to RECURRING PATTERN - a fact that appeared once as a marginal anomaly appears again, possibly with different actors, signalling a structural dynamic rather than noise.

PRECONDITION to ACTIVATION - a structural dependency, vacancy, constraint, or capability seeded earlier is now being used, triggered, or exploited.

INTERNAL to EXTERNAL - a tension that existed only inside the entity has crossed into the external world: litigation, regulatory attention, public announcements, or third-party actors.

DISCREPANCY to INSTITUTIONALIZATION - a gap between what was stated and what was real has now been formalized, contracted, or locked into the entity's structure.

ESCALATION - an actor conflict, policy contradiction, or structural dysfunction flagged earlier has intensified in scale, visibility, or irreversibility.

The test: does this current fact make an earlier off-narrative fact look like step N of something, where N is greater than 1? If yes, it is a weak signal. If the connection is only topical, it is not.

If a development does not fit these modes exactly, name the mode in your own words rather than forcing a fit or discarding the signal.

---

## Step 5 - Group facts when the signal requires it

Some weak signals become visible only when two or more current-period facts are read together, because each alone is unremarkable but jointly they constitute the next step in an earlier tension.

When the signal only exists at the group level:
- Add ONE explanations entry for the whole group.
- Set index to the list of all current-period fact indices in that group.
- Write a single shared theme and explanation.
- Also include every index in weak_signal_indices.

For a single-fact weak signal, use a one-element list.

---

## Strict ignore list

Do NOT flag a current-period fact or group if:
- Its only link to an earlier fact is a shared domain, actor name, or topic, with no shared underlying tension.
- It develops the current period's dominant narrative and merely mentions the same topic as an earlier marginal fact.
- The connection requires inferring facts, causal mechanisms, or relationships not present in the corpus.
- The flag depends on prior knowledge of the company, or of later events, that are not stated in the provided quintuples.
- It restates the same real-world content as another fact with different wording.
- The facts are grouped only because they share an actor or a topic. Grouping is valid only when they jointly constitute a step that none captures alone.

In general: if the only reason to flag a fact is how it is written rather than what it says about the world, do not flag it.

---

## Calibration rules

- Return indices into the CURRENT-PERIOD list only.
- Prefer precision over recall. A plausible thematic connection is not sufficient; the tension must be traceable and the development concrete.
- The earlier baseline matters: a fact is a weak signal only if the thread it continues was off-narrative back then too.
- No speculation, and no knowledge of later periods.
- Reason only from the numbered quintuples in this prompt. Treat the central entity as unknown besides those quintuples.
- Do not group facts merely to manufacture a signal.
- If no current-period fact qualifies, return empty lists.

---

## Structured output (required)

Return a single JSON object with exactly these fields:

### weak_signal_indices (list of integers)
Flat, deduplicated list of every flagged current-period fact index. Must equal the union of all index lists in explanations.

### explanations (list of objects)
One object per weak signal (single fact or group). Each object has:

- index (list of integers): one or more current-period fact indices forming this weak signal. Use a list even for a single fact.
- theme (string): the shared underlying tension linking the earlier period to this signal.
- explanation (string): how this fact or group develops that tension - the mode of development, the earlier source, and the concrete development. For groups, explain why the signal exists only when the facts are read together."""

COT_USER = """Scope: {scope_label}

DTKG quintuples from earlier periods (context only, never flag these):
{past_facts}

Current-period DTKG quintuples (return indices into this list):
{current_facts}

Return structured output with:
- weak_signal_indices: flat deduplicated list of flagged current-period quintuple indices
- explanations: one item per weak signal; each index is a list of current-period quintuple indices sharing the same theme and explanation"""

COT_MEMORY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", COT_SYSTEM), ("human", COT_USER)]
)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    prompt: ChatPromptTemplate
    memory: str  # "local" | "expanding"


ARMS: list[ArmSpec] = [
    ArmSpec("ablation_flat_local", FLAT_LOCAL_PROMPT, "local"),
    ArmSpec("ablation_cot_local", COT_LOCAL_PROMPT, "local"),
    ArmSpec("ablation_flat_facts", FLAT_MEMORY_PROMPT, "expanding"),
    ArmSpec("ablation_cot_facts", COT_MEMORY_PROMPT, "expanding"),
]


# ==========================
# Structured output schema (shared by both LLM arms)
# ==========================
class FactWeakSignalExplanation(BaseModel):
    index: list[int] = Field(default_factory=list)
    theme: str = ""
    explanation: str = ""


class FactWeakSignalResult(BaseModel):
    weak_signal_indices: list[int] = Field(default_factory=list)
    explanations: list[FactWeakSignalExplanation] = Field(default_factory=list)


# ==========================
# DTKG loading (full yearly graphs as numbered quintuples)
# ==========================
@dataclass(frozen=True)
class FactRow:
    index: int
    year: int
    text: str


@dataclass
class Chunk:
    """One LLM call: quintuples of `target_rows` are flaggable, `past_rows` are context."""

    label: str
    target_rows: list[FactRow] = field(default_factory=list)
    past_rows: list[FactRow] = field(default_factory=list)


def _unwrap_pickled_snapshot(obj):
    if isinstance(obj, dict) and "knowledge_graph" in obj:
        return obj["knowledge_graph"]
    return obj


def _parse_time_values(values) -> list[datetime]:
    dates: list[datetime] = []
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
    return dates


def _format_bound(values, pick: str) -> str:
    dates = _parse_time_values(values)
    if not dates:
        return ""
    chosen = min(dates) if pick == "start" else max(dates)
    return chosen.strftime("%Y-%m-%d")


def format_quintuple(rel) -> str:
    start = rel.startEntity
    end = rel.endEntity
    t_start = _format_bound(getattr(rel.properties, "t_start", []) or [], "start")
    t_end = _format_bound(getattr(rel.properties, "t_end", []) or [], "end")
    s_label = start.label or "entity"
    o_label = end.label or "entity"
    return (
        f"({start.name}: {s_label}) --> {rel.name} ({t_start}, {t_end}) --> "
        f"({end.name}: {o_label})"
    )


def load_dtkg_rows(snapshots_dir: Path) -> list[FactRow]:
    paths = sorted(
        (path for path in snapshots_dir.glob("*.pkl") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    rows: list[FactRow] = []
    for path in paths:
        year = int(path.stem)
        if not (YEAR_START <= year <= YEAR_END):
            continue
        with open(path, "rb") as handle:
            kg = _unwrap_pickled_snapshot(pickle.load(handle))
        for rel in getattr(kg, "relationships", []):
            text = format_quintuple(rel)
            if text.strip():
                rows.append(FactRow(index=len(rows), year=year, text=text))
    if not rows:
        raise ValueError(f"No DTKG quintuples in {snapshots_dir} for {YEAR_START}-{YEAR_END}")
    return rows


def format_fact_block(rows: list[FactRow]) -> str:
    if not rows:
        return "(none)"
    lines: list[str] = []
    current_year: int | None = None
    for row in rows:
        if row.year != current_year:
            current_year = row.year
            if lines:
                lines.append("")
            lines.append(f"### {current_year}")
        lines.append(f"[{row.index}] {row.text}")
    return "\n".join(lines)


def assert_no_leakage(*blocks: str) -> None:
    """Fail loudly if benchmark ground truth ever reaches the LLM."""
    for block in blocks:
        found = [token for token in FORBIDDEN_IN_FACT_BLOCK if token in block]
        if found:
            raise AssertionError(
                f"Ground-truth leakage in fact block: {found}. "
                "Only DTKG quintuple surfaces may be shown to the LLM."
            )


def target_years_for(rows: list[FactRow]) -> list[int]:
    years = sorted({row.year for row in rows})
    if CALL_GRANULARITY == "all_years":
        return years
    if SKIP_FIRST_YEAR_AS_TARGET and len(years) > 1:
        return years[1:]
    return years


def build_chunks(rows: list[FactRow], memory: str) -> list[Chunk]:
    years = sorted({row.year for row in rows})

    if CALL_GRANULARITY == "all_years":
        return [Chunk(label=f"{years[0]}-{years[-1]}", target_rows=rows, past_rows=[])]

    targets = target_years_for(rows)
    chunks: list[Chunk] = []
    for year in targets:
        target_rows = [r for r in rows if r.year == year]
        if memory == "local":
            past_rows: list[FactRow] = []
        elif memory == "expanding":
            past_rows = [r for r in rows if r.year < year]
        else:
            raise ValueError(f"Unsupported memory mode: {memory}")
        chunks.append(Chunk(label=str(year), target_rows=target_rows, past_rows=past_rows))
    return chunks


# ==========================
# LLM plumbing
# ==========================
def _model_supports_temperature(model_name: str) -> bool:
    name = model_name.lower().strip()
    return not any(name.startswith(prefix) for prefix in _REASONING_MODEL_PREFIXES)


def build_parser():
    if not openai_api_key:
        raise RuntimeError("openai_api_key is empty in api_keys.py")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser

    llm_kwargs = {
        "api_key": openai_api_key,
        "model": OPENAI_MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "timeout": TIMEOUT,
        "max_retries": MAX_RETRIES,
    }
    if _model_supports_temperature(OPENAI_MODEL_NAME):
        llm_kwargs["temperature"] = TEMPERATURE
    else:
        print(f"Model {OPENAI_MODEL_NAME!r} does not support temperature — omitting it.")

    return LangchainOutputParser(
        llm_model=ChatOpenAI(**llm_kwargs),
        embeddings_model=OpenAIEmbeddings(
            api_key=openai_api_key,
            model=OPENAI_EMBEDDINGS_MODEL,
        ),
    )


def _protocol_key(memory: str) -> str:
    if CALL_GRANULARITY == "all_years":
        return "all_years"
    if memory == "local":
        return "snapshot_local"
    return CALL_GRANULARITY


def _legacy_cache_path(arm: str, memory: str) -> Path:
    return TRACES_DIR / f"raw_{arm}_{_protocol_key(memory)}_{PROMPT_VERSION}.json"


def _cache_path(arm: str, memory: str, run_id: int) -> Path:
    return (
        TRACES_DIR
        / f"raw_{arm}_{_protocol_key(memory)}_{PROMPT_VERSION}_run{run_id}.json"
    )


def _resolve_cache_path(arm: str, memory: str, run_id: int) -> Path | None:
    """Return an existing cache for this (arm, run), including the legacy run-0 file."""
    run_path = _cache_path(arm, memory, run_id)
    if run_path.exists():
        return run_path
    if run_id == 0:
        legacy = _legacy_cache_path(arm, memory)
        if legacy.exists():
            return legacy
    return None


def _chunk_variables(chunk: Chunk, memory: str) -> dict:
    current_block = format_fact_block(chunk.target_rows)
    if memory == "local":
        assert_no_leakage(current_block)
        return {"scope_label": chunk.label, "current_facts": current_block}

    past_block = format_fact_block(chunk.past_rows)
    assert_no_leakage(past_block, current_block)
    return {
        "scope_label": chunk.label,
        "past_facts": past_block,
        "current_facts": current_block,
    }


def _save_trace(
    arm: str,
    chunk: Chunk,
    prompt: ChatPromptTemplate,
    variables: dict,
    result: FactWeakSignalResult,
    run_id: int = 0,
) -> None:
    path = TRACES_DIR / "dtkg" / f"{arm}_run{run_id}_{chunk.label}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        f"=== {message.type.upper()} ===\n{message.content}"
        for message in prompt.format_messages(**variables)
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== PROMPT ===\n")
        f.write("\n\n".join(sections))
        f.write("\n\n=== RESPONSE ===\n")
        f.write(json.dumps(result.model_dump(), indent=2))
        f.write("\n")


async def run_arm(
    parser,
    spec: ArmSpec,
    chunks: list[Chunk],
    run_id: int = 0,
) -> list[FactWeakSignalResult]:
    cache = None if FORCE_RERUN else _resolve_cache_path(spec.name, spec.memory, run_id)
    if cache is not None:
        print(f"[{spec.name} run={run_id}] reusing cached LLM output: {cache.name}")
        with open(cache, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(chunks):
            return [FactWeakSignalResult(**item) for item in cached]
        print(
            f"[{spec.name} run={run_id}] cache has {len(cached)} chunk(s), "
            f"expected {len(chunks)} — rerunning."
        )

    if parser is None:
        raise RuntimeError(
            f"[{spec.name} run={run_id}] no usable cache and no LLM parser available"
        )

    variables_list = [_chunk_variables(chunk, spec.memory) for chunk in chunks]

    print(
        f"[{spec.name} run={run_id}] calling {OPENAI_MODEL_NAME} "
        f"on {len(variables_list)} chunk(s)..."
    )
    raw = await parser.batch_structured_chat_calls(
        prompt_template=spec.prompt,
        variables_list=variables_list,
        output_data_structure=FactWeakSignalResult,
    )

    results: list[FactWeakSignalResult] = []
    for item in raw:
        results.append(item if isinstance(item, FactWeakSignalResult) else FactWeakSignalResult())

    out_cache = _cache_path(spec.name, spec.memory, run_id)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(out_cache, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
        f.write("\n")

    for chunk, variables, result in zip(chunks, variables_list, results):
        try:
            _save_trace(spec.name, chunk, spec.prompt, variables, result, run_id=run_id)
        except Exception as exc:
            print(f"[{spec.name} run={run_id}] warning: failed to save trace for {chunk.label}: {exc}")
    return results


# ==========================
# Detections
# ==========================
def _normalize_index_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    out: list[int] = []
    for item in value:
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, (list, tuple)):
            out.extend(int(sub) for sub in item)
    return out


def build_description(fact_texts: list[str], theme: str, explanation: str) -> str:
    """Same tokenization as c_unseen_runner._build_description, for fair anchoring."""
    parts: list[str] = []
    for text in fact_texts:
        parts.extend(_TOKEN_RE.split(text.lower()))
    for extra in (theme or "", explanation or ""):
        extra = extra.strip()
        if extra:
            parts.extend(_TOKEN_RE.split(extra.lower()))
    return " ".join(token for token in parts if token)


def flagged_indices(result: FactWeakSignalResult, allowed: set[int]) -> set[int]:
    """Every in-protocol index the model flagged, however it reported it."""
    found = set(_normalize_index_list(result.weak_signal_indices))
    for entry in result.explanations:
        found.update(_normalize_index_list(entry.index))
    return found & allowed


def results_to_detections(
    chunks: list[Chunk],
    results: list[FactWeakSignalResult],
    rows: list[FactRow],
) -> list[dict]:
    by_index = {row.index: row for row in rows}
    detections: list[dict] = []
    out_of_protocol = 0

    for chunk, result in zip(chunks, results):
        allowed = {row.index for row in chunk.target_rows}
        groups: list[tuple[list[int], str, str]] = []
        explained: set[int] = set()

        for entry in result.explanations:
            raw_indices = set(_normalize_index_list(entry.index))
            indices = sorted(raw_indices & allowed)
            out_of_protocol += len(raw_indices - allowed)
            if indices:
                groups.append((indices, entry.theme, entry.explanation))
                explained.update(indices)

        if not explained or EMIT_UNEXPLAINED_INDICES:
            for idx in sorted(set(_normalize_index_list(result.weak_signal_indices))):
                if idx in allowed and idx not in explained:
                    groups.append(([idx], "", ""))

        for group_indices, theme, explanation in groups:
            years = [by_index[i].year for i in group_indices]
            year = max(years) if DETECTION_YEAR_RULE == "max" else min(years)
            fact_texts = [by_index[i].text for i in group_indices]
            detections.append(
                {
                    "year": year,
                    "description": build_description(fact_texts, theme, explanation),
                    "scope_label": chunk.label,
                    "theme": theme,
                    "explanation": explanation,
                    "fact_indices": group_indices,
                    "fact_years": years,
                    "facts": fact_texts,
                }
            )

    if out_of_protocol:
        print(
            f"        note: dropped {out_of_protocol} out-of-protocol index reference(s) "
            "(context or invalid facts)"
        )

    detections.sort(key=lambda row: (row["year"], row["fact_indices"]))
    for bucket_id, detection in enumerate(detections):
        detection["bucket_id"] = bucket_id
    return detections


def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def print_arm_summary(arm: str, detections: list[dict], run_id: int | None = None) -> None:
    by_year: dict[int, int] = {}
    for detection in detections:
        by_year[detection["year"]] = by_year.get(detection["year"], 0) + 1
    grouped = sum(1 for d in detections if len(d.get("fact_indices", [])) > 1)
    prefix = f"[{arm} run={run_id}]" if run_id is not None else f"[{arm}]"
    print(f"{prefix} {len(detections)} detection(s), {grouped} multi-fact group(s)")
    print(
        "        per year: "
        + (", ".join(f"{year}:{by_year[year]}" for year in sorted(by_year)) or "none")
    )


# ==========================
# C-Unseen replicates
# ==========================
_C_UNSEEN_STATE: dict = {}
C_UNSEEN_RUN_REPORTS: Path = TRACES_DIR / "c_unseen_quintuple_runs_grounded"


def _c_unseen_detections_path(run_id: int) -> Path:
    return ABLATION_DIR / f"detections_c_unseen_run{run_id}.json"


def _load_c_unseen_runner():
    if "cu" in _C_UNSEEN_STATE:
        return _C_UNSEEN_STATE["cu"], _C_UNSEEN_STATE["run_script"]

    spec = importlib.util.spec_from_file_location(
        "c_unseen_runner_mod",
        DETECTORS / "c_unseen_runner.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load c_unseen_runner.py")
    cu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cu)
    run_script = cu._load_run_script()
    _C_UNSEEN_STATE["cu"] = cu
    _C_UNSEEN_STATE["run_script"] = run_script
    return cu, run_script


async def run_c_unseen_arm(run_id: int) -> list[dict]:
    """One C-Unseen replicate. Run 0 reuses detections_c_unseen.json when present."""
    out = _c_unseen_detections_path(run_id)
    if not FORCE_RERUN and out.exists():
        print(f"[c_unseen run={run_id}] reusing cached detections: {out.name}")
        with open(out, "r", encoding="utf-8") as f:
            return json.load(f)

    if not FORCE_RERUN and run_id == 0 and C_UNSEEN_DETECTIONS.exists():
        print(f"[c_unseen run={run_id}] reusing {C_UNSEEN_DETECTIONS.name}")
        with open(C_UNSEEN_DETECTIONS, "r", encoding="utf-8") as f:
            detections = json.load(f)
        save_json(detections, out)
        return detections

    cu, run_script = _load_c_unseen_runner()
    reports = C_UNSEEN_RUN_REPORTS / f"run{run_id}"
    reports.mkdir(parents=True, exist_ok=True)
    cu.REPORTS_DIR = reports
    cu._configure_run_script(run_script)

    snapshot_paths = run_script._get_snapshot_paths()
    labels = [run_script._snapshot_label_from_path(p) for p in snapshot_paths]
    print(
        f"[c_unseen run={run_id}] running pipeline on {len(snapshot_paths)} "
        f"snapshot(s); traces -> {reports}"
    )
    kgs = cu._load_kgs(run_script, snapshot_paths)
    traces_complete = cu._traces_complete(run_script, labels)
    kgs = await cu._maybe_run_c_unseen(run_script, kgs, labels, traces_complete)
    detections = cu._extract_detections(run_script, kgs, labels)
    save_json(detections, out)
    print(f"[c_unseen run={run_id}] wrote {out} ({len(detections)} detections)")
    return detections


# ==========================
# Scoring (delegated to scorer.py)
# ==========================
def load_scorer():
    scorer_path = SCORING / "scorer.py"
    spec = importlib.util.spec_from_file_location("weak_signal_scorer", scorer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["weak_signal_scorer"] = module
    spec.loader.exec_module(module)
    return module


SCALAR_METRIC_KEYS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "detections_total",
    "true_positives",
    "events_covered",
    "mean_lead_days",
    "mean_lead_years",
)

DELTA_STEPS: list[tuple[str, str, str]] = [
    ("local flat -> memory flat", "ablation_flat_local", "ablation_flat_facts"),
    ("local cot -> memory cot", "ablation_cot_local", "ablation_cot_facts"),
    ("local flat -> local cot", "ablation_flat_local", "ablation_cot_local"),
    ("memory flat -> memory cot", "ablation_flat_facts", "ablation_cot_facts"),
    ("memory cot -> cot+dtkg", "ablation_cot_facts", "c_unseen"),
    ("local cot -> cot+dtkg", "ablation_cot_local", "c_unseen"),
]


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(var)


def _fmt_ms(mean: float, std: float, digits: int = 3) -> str:
    if std == 0.0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def load_scoring_context(scorer):
    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    return (
        scorer.compute_windows(benchmark),
        scorer.load_anchors(scorer.ANCHORS_JSON),
        scorer.load_overrides(scorer.OVERRIDES_JSON),
    )


def score_arm_files(
    scorer,
    arm_files: list[tuple[str, Path]],
    signals_meta,
    anchors,
    overrides,
) -> dict[str, dict]:
    results_by_method: dict[str, dict] = {}
    for method, path in arm_files:
        results_by_method[method] = scorer.score_file(
            method, path, signals_meta, anchors, overrides
        )
    return results_by_method


def aggregate_runs(
    runs_by_method: dict[str, list[dict]],
    k_levels: list[int],
) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for method, runs in runs_by_method.items():
        per_k: dict[str, dict] = {"n": len(runs)}
        for k in k_levels:
            key = f"k={k}"
            metrics: dict[str, dict] = {}
            for metric in SCALAR_METRIC_KEYS:
                values = [float(run[key][metric]) for run in runs]
                mean, std = mean_std(values)
                metrics[metric] = {
                    "mean": mean,
                    "std": std,
                    "values": values,
                }
            per_k[key] = metrics
        summary[method] = per_k
    return summary


def print_aggregate_tables(summary: dict[str, dict], k_levels: list[int]) -> None:
    header = (
        f"{'method':<22} | {'n':>3} | {'detections':>14} | {'precision':>13} | "
        f"{'recall':>13} | {'f1':>13} | {'events':>10} | {'lead_yrs':>12}"
    )
    for k in k_levels:
        print(f"\n--- k = {k}  (mean ± sample std over RUN_TIMES={RUN_TIMES}) ---")
        print(header)
        print("-" * len(header))
        for method in sorted(summary):
            row = summary[method][f"k={k}"]
            n = summary[method]["n"]
            print(
                f"{method:<22} | {n:>3} | "
                f"{_fmt_ms(row['detections_total']['mean'], row['detections_total']['std'], 1):>14} | "
                f"{_fmt_ms(row['precision']['mean'], row['precision']['std']):>13} | "
                f"{_fmt_ms(row['recall']['mean'], row['recall']['std']):>13} | "
                f"{_fmt_ms(row['f1']['mean'], row['f1']['std']):>13} | "
                f"{_fmt_ms(row['events_covered']['mean'], row['events_covered']['std'], 2):>10} | "
                f"{_fmt_ms(row['mean_lead_years']['mean'], row['mean_lead_years']['std'], 2):>12}"
            )


def print_aggregate_deltas(
    runs_by_method: dict[str, list[dict]],
    k_levels: list[int],
) -> None:
    print("\nAblation deltas (mean ± std of per-run differences):")
    for k in k_levels:
        print(f"\n--- k = {k} ---")
        key = f"k={k}"
        for label, left, right in DELTA_STEPS:
            if left not in runs_by_method or right not in runs_by_method:
                continue
            left_runs = runs_by_method[left]
            right_runs = runs_by_method[right]
            n = min(len(left_runs), len(right_runs))
            # If one arm is a single existing run (C-Unseen), compare every
            # left replicate against that constant.
            if len(left_runs) != len(right_runs):
                if len(right_runs) == 1:
                    n = len(left_runs)
                    f1s = [right_runs[0][key]["f1"] - r[key]["f1"] for r in left_runs]
                    recs = [
                        right_runs[0][key]["recall"] - r[key]["recall"] for r in left_runs
                    ]
                    leads = [
                        right_runs[0][key]["mean_lead_years"] - r[key]["mean_lead_years"]
                        for r in left_runs
                    ]
                elif len(left_runs) == 1:
                    n = len(right_runs)
                    f1s = [r[key]["f1"] - left_runs[0][key]["f1"] for r in right_runs]
                    recs = [
                        r[key]["recall"] - left_runs[0][key]["recall"] for r in right_runs
                    ]
                    leads = [
                        r[key]["mean_lead_years"] - left_runs[0][key]["mean_lead_years"]
                        for r in right_runs
                    ]
                else:
                    f1s = [
                        right_runs[i][key]["f1"] - left_runs[i][key]["f1"]
                        for i in range(n)
                    ]
                    recs = [
                        right_runs[i][key]["recall"] - left_runs[i][key]["recall"]
                        for i in range(n)
                    ]
                    leads = [
                        right_runs[i][key]["mean_lead_years"]
                        - left_runs[i][key]["mean_lead_years"]
                        for i in range(n)
                    ]
            else:
                f1s = [
                    right_runs[i][key]["f1"] - left_runs[i][key]["f1"] for i in range(n)
                ]
                recs = [
                    right_runs[i][key]["recall"] - left_runs[i][key]["recall"]
                    for i in range(n)
                ]
                leads = [
                    right_runs[i][key]["mean_lead_years"]
                    - left_runs[i][key]["mean_lead_years"]
                    for i in range(n)
                ]
            f1_m, f1_s = mean_std(f1s)
            rec_m, rec_s = mean_std(recs)
            lead_m, lead_s = mean_std(leads)
            print(
                f"  {label:<28} n={n}  "
                f"Δf1 {_fmt_ms(f1_m, f1_s)}   "
                f"Δrecall {_fmt_ms(rec_m, rec_s)}   "
                f"Δlead {_fmt_ms(lead_m, lead_s, 2)}"
            )


def compute_fact_level_row(
    results: list[FactWeakSignalResult],
    chunks: list[Chunk],
    truth: set[int],
) -> dict[str, float]:
    targetable = {row.index for chunk in chunks for row in chunk.target_rows}
    reachable = truth & targetable
    flagged: set[int] = set()
    for chunk, result in zip(chunks, results):
        flagged |= flagged_indices(result, {r.index for r in chunk.target_rows})
    tp = len(flagged & reachable)
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / len(reachable) if reachable else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    share = len(flagged) / len(targetable) if targetable else 0.0
    return {
        "flagged": float(len(flagged)),
        "targetable": float(len(targetable)),
        "reachable": float(len(reachable)),
        "tp": float(tp),
        "share": share,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_fact_level_diagnostic(
    runs_by_arm: dict[str, list[list[FactWeakSignalResult]]],
    chunks_by_arm: dict[str, list[Chunk]],
    rows: list[FactRow],
) -> dict[str, dict]:
    """
    Fact-level check against the benchmark's own role == "weak_signal" labels.

    The shared scorer measures precision per *detection group*, so an arm that
    flags hundreds of facts but bundles them into a few explained groups scores
    as precise. This counts flagged facts directly and exposes over-flagging.
    Ground-truth roles are read here only — never in a prompt (assert_no_leakage).

    C-Unseen is absent by construction: it flags KG triples, not fact indices.
    """
    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    roles: list[str] = []
    for year_key in sorted(benchmark["facts"], key=int):
        year = int(year_key)
        if not (YEAR_START <= year <= YEAR_END):
            continue
        for fact in benchmark["facts"][year_key]:
            if str(fact["text"]).strip():
                roles.append(str(fact.get("role", "")))

    if len(roles) != len(rows):
        print("\nFact-level diagnostic skipped: fact alignment mismatch.")
        return {}

    truth = {i for i, role in enumerate(roles) if role == "weak_signal"}
    summary: dict[str, dict] = {}
    print("\nFact-level diagnostic vs benchmark weak_signal labels (mean ± std):")
    for arm, runs in runs_by_arm.items():
        chunks = chunks_by_arm[arm]
        rows_out = [compute_fact_level_row(run, chunks, truth) for run in runs]
        metric_names = ["flagged", "share", "tp", "precision", "recall", "f1"]
        agg = {}
        for metric in metric_names:
            mean, std = mean_std([row[metric] for row in rows_out])
            agg[metric] = {
                "mean": mean,
                "std": std,
                "values": [row[metric] for row in rows_out],
            }
        agg["n"] = len(runs)
        agg["targetable"] = rows_out[0]["targetable"] if rows_out else 0.0
        agg["reachable"] = rows_out[0]["reachable"] if rows_out else 0.0
        summary[arm] = agg
        print(
            f"  {arm:<22} n={len(runs)}  "
            f"flagged {_fmt_ms(agg['flagged']['mean'], agg['flagged']['std'], 1)}"
            f" / {int(agg['targetable'])} "
            f"({_fmt_ms(agg['share']['mean'] * 100, agg['share']['std'] * 100, 1)}%)  "
            f"tp {_fmt_ms(agg['tp']['mean'], agg['tp']['std'], 1)}  "
            f"precision {_fmt_ms(agg['precision']['mean'], agg['precision']['std'])}  "
            f"recall {_fmt_ms(agg['recall']['mean'], agg['recall']['std'])}  "
            f"f1 {_fmt_ms(agg['f1']['mean'], agg['f1']['std'])}"
        )
    return summary


def print_ablation_deltas(results_by_method: dict[str, dict], k_levels: list[int]) -> None:
    print("\nAblation deltas (single run):")
    for k in k_levels:
        print(f"\n--- k = {k} ---")
        for label, left, right in DELTA_STEPS:
            if left not in results_by_method or right not in results_by_method:
                continue
            a = results_by_method[left][f"k={k}"]
            b = results_by_method[right][f"k={k}"]
            print(
                f"  {label:<28} "
                f"f1 {a['f1']:.3f} -> {b['f1']:.3f} ({b['f1'] - a['f1']:+.3f})   "
                f"recall {a['recall']:.3f} -> {b['recall']:.3f} ({b['recall'] - a['recall']:+.3f})   "
                f"lead {a['mean_lead_years']:.2f} -> {b['mean_lead_years']:.2f} "
                f"({b['mean_lead_years'] - a['mean_lead_years']:+.2f})"
            )


async def main_async() -> None:
    if RUN_TIMES < 1:
        raise ValueError(f"RUN_TIMES must be >= 1, got {RUN_TIMES}")

    LLM_ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_dtkg_rows(SNAPSHOTS_DIR)
    print(f"Loaded {len(rows)} DTKG quintuples ({YEAR_START}-{YEAR_END}) from {SNAPSHOTS_DIR}")
    print(f"Target years: {target_years_for(rows)}")
    print(
        "Fields shown to the LLM: numbered quintuples "
        "(subject:type) --> predicate (t_start, t_end) --> (object:type)"
    )
    if rows:
        print("Sample quintuple:", rows[0].text)
    print(f"Model: {OPENAI_MODEL_NAME}")
    print(f"Replicates: RUN_TIMES={RUN_TIMES}")
    if CALL_GRANULARITY == "all_years":
        print("  WARNING: all_years lets the model see the future. Hindsight oracle,")
        print("           not a fair comparison against C-Unseen.")

    chunks_by_memory: dict[str, list[Chunk]] = {}
    for memory in ("local", "expanding"):
        chunks = build_chunks(rows, memory)
        chunks_by_memory[memory] = chunks
        print(f"\nProtocol {memory}: {len(chunks)} call(s) per arm per run")
        for chunk in chunks:
            print(
                f"  target {chunk.label}: {len(chunk.target_rows):>3} flaggable, "
                f"{len(chunk.past_rows):>3} context"
            )

    needs_llm = any(
        FORCE_RERUN or _resolve_cache_path(spec.name, spec.memory, run_id) is None
        for spec in ARMS
        for run_id in range(RUN_TIMES)
    )
    parser = build_parser() if needs_llm else None

    scorer = load_scorer()
    signals_meta, anchors, overrides = load_scoring_context(scorer)
    runs_by_method: dict[str, list[dict]] = {spec.name: [] for spec in ARMS}
    runs_by_method["c_unseen"] = []
    llm_runs_by_arm: dict[str, list[list[FactWeakSignalResult]]] = {
        spec.name: [] for spec in ARMS
    }
    chunks_by_arm: dict[str, list[Chunk]] = {
        spec.name: chunks_by_memory[spec.memory] for spec in ARMS
    }

    for run_id in range(RUN_TIMES):
        print(f"\n{'=' * 50}")
        print(f"  RUN {run_id + 1} / {RUN_TIMES}")
        print("=" * 50)
        arm_files: list[tuple[str, Path]] = []
        for spec in ARMS:
            chunks = chunks_by_arm[spec.name]
            results = await run_arm(parser, spec, chunks, run_id=run_id)
            llm_runs_by_arm[spec.name].append(results)
            detections = results_to_detections(chunks, results, rows)
            path = LLM_ABLATION_DIR / f"detections_{spec.name}_run{run_id}.json"
            save_json(detections, path)
            print_arm_summary(spec.name, detections, run_id=run_id)
            print(f"        wrote {path}")
            arm_files.append((spec.name, path))

        c_unseen_detections = await run_c_unseen_arm(run_id)
        c_unseen_path = _c_unseen_detections_path(run_id)
        print_arm_summary("c_unseen", c_unseen_detections, run_id=run_id)
        arm_files.append(("c_unseen", c_unseen_path))

        scored = score_arm_files(scorer, arm_files, signals_meta, anchors, overrides)
        for method, results in scored.items():
            runs_by_method[method].append(results)
            save_json(results, LLM_ABLATION_DIR / f"results_{method}_run{run_id}.json")

        compact = "  ".join(
            f"{method} f1@k2={scored[method]['k=2']['f1']:.3f}"
            for method in scored
        )
        print(f"  {compact}")

    summary = aggregate_runs(runs_by_method, scorer.K_LEVELS)
    print_aggregate_tables(summary, scorer.K_LEVELS)
    print_aggregate_deltas(runs_by_method, scorer.K_LEVELS)
    fact_summary = print_fact_level_diagnostic(llm_runs_by_arm, chunks_by_arm, rows)

    payload = {
        "run_times": RUN_TIMES,
        "model": OPENAI_MODEL_NAME,
        "arms": list(runs_by_method.keys()),
        "scorer": summary,
        "fact_level": fact_summary,
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
    summary_path = LLM_ABLATION_DIR / "results_summary.json"
    save_json(payload, summary_path)
    print(f"\nWrote aggregate summary: {summary_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    print("=" * 50)
    print("  C-UNSEEN ABLATION: LOCAL / MEMORY / DTKG")
    print("=" * 50)
    main()
