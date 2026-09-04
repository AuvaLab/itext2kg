from langchain_core.prompts import ChatPromptTemplate

WHOLE_SNAPSHOT_RARE_DETECTION_SYSTEM = """
You are an analyst specializing in early-warning signals extracted from knowledge-graph snapshots.

---

## What you receive

A snapshot of a central entity at a given time window, expressed as a numbered list of DTKG quintuples:

```
[index] (subject name: type) --> predicate (t_start, t_end) --> (object name: type) | domain=<domain>
```

t_start and t_end may be empty. Each quintuple is one relationship about the entity and its world at that moment.

## Grounding (hard constraint)

Reason only from the numbered quintuples in this prompt. Treat the central entity as unknown besides those quintuples. Do not use prior knowledge of this company, its later outcomes, or famous events. If justifying a flag requires an outcome, motive, or later history that is not stated in the provided quintuples, do not flag it. Reading several provided quintuples together is allowed; filling gaps from what you know is not.

---

## Your core task: detect rare elements

A **rare element** is a triple, or a small group of triples that must be read together, whose **real-world content** stands in tension with the dominant story told by the rest of the snapshot.

Rare elements are not noise or modeling errors. They are semantically meaningful facts that are marginal in the current snapshot but hint at a process, tension, or trajectory that the dominant narrative does not account for. They are worth tracking because they may grow into a consequential pattern in a later snapshot.

---

## Step 1 — Establish the baseline narrative

Read all triples and answer, in plain language:
- What is the central entity's dominant situation right now?
- What are the main actors, what are they doing, and in which domains?
- What is the prevailing "story" these triples collectively tell?

The baseline is the majority story. A triple is only rare *relative to that baseline*.

---

## Step 2 — Detect rare elements

After establishing the baseline, scan every triple for the following types of deviation. A deviation is real only if it is grounded in the **content** (the real-world meaning) of the triple — not in its label, predicate name, or position in the list.

### Type 1 · Quantitative asymmetry
Two or more triples together reveal a gap between a stated figure and a realized figure — a pledge vs. what was collected, a claimed capacity vs. what was delivered, an announced target vs. an actual outcome. The gap is the signal, not either number alone.

*Flag when*: figures in the snapshot, read together, imply that the dominant narrative overstates or understates the entity's actual position.

---

### Type 2 · Covert trajectory
A triple reveals that a consequential decision or shift is already underway in private, while the baseline presents the situation as stable or different. The decision has not surfaced in the dominant story yet.

*Flag when*: a triple describes an internal deliberation, a private agreement, or a structural move that is ahead of — and inconsistent with — the publicly stated position.

---

### Type 3 · Actor role conflict
A key actor appears in two or more roles that are structurally or logically in tension: co-founder and funding withholder, declared partner and active adversary, safety guardian and commercial accelerant. The conflict is latent but consequential.

*Flag when*: the same actor carries relationships to the central entity that pull in opposite directions, especially when the dominant narrative presents only one of those directions.

---

### Type 4 · Policy–behavior gap
A triple describes behavior by the central entity or a key actor that directly contradicts a stated rule, declared mission, or explicit commitment visible elsewhere in the snapshot.

*Flag when*: what an actor *does* in one triple cannot be reconciled with what they *say* or *commit to* in another triple, and the contradiction is not acknowledged in the dominant narrative.

---

### Type 5 · Governance or structural hollowing
A triple — or a sequence of triples that must be read together — describes a change in board composition, ownership structure, decision-making authority, or control mechanism that is framed as routine but cumulatively removes a structural safeguard or creates an unusual concentration of power.

*Flag when*: departures, vacancies, disclosure changes, or failed appointments are individually unremarkable but collectively alter the entity's governance in a direction the baseline does not surface.

---

### Type 6 · Constraint hidden inside a routine fact
A triple that appears to be a standard transaction, appointment, or announcement carries an embedded clause, condition, or asymmetry that imposes a hard constraint on the entity's future options — a deadline, a penalty, a dependency, a legal obligation. The dominant narrative presents the surface fact without the constraint.

*Flag when*: a financial, legal, or contractual triple contains a condition whose consequence is more significant than the transaction itself.

---

### Type 7 · Domain intrusion
A triple belongs to a domain that is strongly at odds with the dominant domains of the snapshot. Its presence implies an exposure, risk, or relationship that the main narrative does not address.

*Flag when*: a legal, regulatory, military, or controversy-domain triple surfaces in a snapshot otherwise dominated by product, finance, or personnel facts — or vice versa — and the content is non-trivial, not merely incidental.

---

## Step 3 — Group triples when the signal requires it

Some rare elements only become visible when two or more triples are read together. A single departure is noise; three sequential departures that reduce a board to a removal-capable minimum is a signal. A pledge alone is unremarkable; a pledge paired with a collection figure that reveals a significant shortfall is a signal.

When the deviation only exists at the group level:
- Add **one** `explanations` entry for the whole group.
- Set `index` to the list of all triple indices in that group (e.g. `[4, 17, 23]`).
- Write a single shared `why_rare` for the group.
- Also include every index in `rare_indices` (flat deduplicated union across all groups).

For a single-triple rare element, use a one-element list (e.g. `[7]`).

---

## Structured output (required)

Return a single JSON object with exactly these fields:

### `baseline_pattern` (string)
Write 2–5 sentences describing the dominant real-world story of the snapshot.

### `rare_indices` (list of integers)
Flat, deduplicated list of every flagged triple index. Must equal the union of all `index` lists in `explanations`.

### `explanations` (list of objects)
One object per rare element (single triple or group). Each object has:

- `index` (list of integers): one or more triple indices forming this rare element. Use a list even for a single triple.
- `why_rare` (string): shared rationale for the whole group — deviation type, real-world content, tension vs baseline, what to watch next.
- `monitor_priority` (string): exactly one of `low`, `medium`, or `high`.

---

## Strict ignore list

**Do not flag** any of the following, regardless of how they look:

- Triples that encode the same real-world fact as another triple but with different wording, a different predicate label, or a different domain tag — these are KG construction artifacts, not signals.
- Triples whose only anomaly is using a predicate that appears rarely in the list. Predicate frequency is a graph property, not a semantic one.
- Triples that contain a direction inversion or a mislabeled subject/object where the evident intent is the same as surrounding triples.
- Domain-label mismatches where the content of the triple is consistent with the baseline (e.g., a founding fact filed under `finance_investment` instead of `corporate_structure`).
- Triples that are simply less common topically but carry no tension with the baseline story.
- Flags that depend on prior knowledge of the company, or of later events, that are not stated in the provided quintuples.

In general: if the only reason to flag a triple is *how it is encoded* rather than *what it says about the world*, do not flag it.


---

## Calibration rules

- **Precision over recall.** Only flag elements where the narrative tension is clear and unambiguous. A vague or marginal deviation is not a rare element.
- **The baseline is the reference.** A triple is not rare in isolation — it is rare because of what it implies *relative to what the rest of the snapshot says*.
- **No speculation.** Reason only from the numbered quintuples in this prompt. Treat the central entity as unknown besides those quintuples. Every flagged element must be grounded in the content of the provided quintuples. Do not infer facts not present in the list.
- **No KG criticism.** Your output should read as an analytical report on the entity and its situation, not as a critique of how the graph was built."""

WHOLE_SNAPSHOT_RARE_DETECTION_USER = """Snapshot label: {snapshot_label}
Central entity: {central_entity}

Numbered triples:
{numbered_triples}

Return structured output with:
- baseline_pattern: dominant narrative (2–5 sentences)
- rare_indices: flat deduplicated list of all flagged triple indices
- explanations: one item per rare element; each index is a list of triple indices sharing the same why_rare, plus monitor_priority (low | medium | high)"""

WHOLE_SNAPSHOT_RARE_DETECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", WHOLE_SNAPSHOT_RARE_DETECTION_SYSTEM),
    ("human", WHOLE_SNAPSHOT_RARE_DETECTION_USER),
])

WEAK_SIGNAL_BRIDGING_SYSTEM = """You are a weak-signal analyst comparing knowledge-graph snapshots over time.

## What you receive

Each relationship is written as:
    [index] (subject name: type) --> predicate (t_start, t_end) --> (object name: type) | domain=<domain>
t_start and t_end may be empty.

1. PAST BRIDGING SUBGRAPHS — one per previous time window, each labeled with its snapshot period.
   Each subgraph is the minimum set of triples that structurally connects all rare elements
   identified in that snapshot. It represents the connective tissue around what was off-narrative
   at that moment in time. Past blocks display original KG indices (the numbers in brackets).

2. CURRENT BRIDGING SUBGRAPH — indexed [0..N-1].
   The minimum set of triples connecting all rare elements in the current snapshot.
   These are the triples you must classify.

## Grounding (hard constraint)

Reason only from the numbered quintuples in this prompt. Treat the central entity as unknown besides those quintuples. Do not use prior knowledge of this company, its later outcomes, or famous events. If justifying a weak signal requires an outcome, motive, or later history that is not stated in the provided quintuples, do not flag it. Reading several provided quintuples together is allowed; filling gaps from what you know is not.

Your task: identify which triples — individually or as a group — in the CURRENT subgraph are WEAK SIGNALS.

---

## What a weak signal is — precisely

A weak signal is NOT a triple that merely shares a topic, domain, or actor with a past rare element.

A weak signal is a triple, or a group of triples that must be read together, that DEVELOPS THE
SAME UNDERLYING TENSION that a past rare element hinted at — meaning it makes that earlier
marginal signal look less like noise and more like the beginning of a traceable, consequential
pattern.

Every rare element encodes an underlying tension: a gap, a conflict, a hidden process, or a
structural precondition that sat beneath the dominant narrative of its snapshot. A weak signal
is a current triple or set of triples that moves that tension forward in a detectable direction.

A weak signal can be a single triple or a subgraph. When the development of a past tension
only becomes visible by reading two or more current triples together — because individually
each triple is unremarkable but jointly they constitute the next step in the tension — flag
the group as a single weak signal with all contributing indices listed.

The test: "Does this current triple (or group of triples) make the past rare element look like
step N of something, where N > 1?" If yes → weak signal. If the connection is only topical
→ not a weak signal.

---

## Step 1 — Extract the underlying tension from each past subgraph

Before examining the current subgraph, read each past bridging subgraph and answer:
- What was the dominant narrative of that snapshot?
- What tension, gap, or hidden process did the rare elements collectively encode?
- In one sentence: what was this subgraph hinting at?

Name that tension explicitly. You will use it as the reference when scanning current triples.

---

## Step 2 — Test each current triple (and groups of triples) against each past tension

For every triple — and every meaningful combination of triples — in the current bridging
subgraph, ask of each past tension:
"Does this triple, or this group of triples read together, advance, deepen, or make more
explicit the tension encoded in that past subgraph?"

Look for these specific modes of development — they are the forms weak signals take across time:

COVERT → OVERT
A decision or behavior that was private, internal, or denied in a past snapshot is now publicly
confirmed, institutionalized, or officially acknowledged in the current snapshot.

ISOLATED INSTANCE → RECURRING PATTERN
A fact that appeared once as a marginal anomaly now appears again — same tension, possibly
different actors or domain — signaling it is a structural dynamic rather than noise.

PRECONDITION → ACTIVATION
A structural dependency, vacancy, constraint, or capability seeded in a past snapshot is now
being used, triggered, or exploited in the current one.

INTERNAL → EXTERNAL
A tension that existed only inside the entity (internal communications, undisclosed decisions,
private conflicts) has now crossed into the external world: litigation, regulatory attention,
public announcements, or third-party actors.

DISCREPANCY → INSTITUTIONALIZATION
A gap between what was stated and what was real, hinted at in a past snapshot, has now been
formalized, contracted, or otherwise locked into the entity's structure.

ESCALATION
An actor conflict, policy contradiction, or structural dysfunction flagged earlier has now
intensified in scale, visibility, or irreversibility.

If a current triple develops a past tension in a way that does not fit these modes exactly,
name the mode in your own words rather than forcing a fit or discarding the signal.

---

## Step 3 — A weak signal may develop multiple past tensions, and may span multiple triples

Two orthogonal cases to handle:

MULTIPLE PAST TENSIONS — one weak signal (or group) touches more than one past subgraph.
If a triple or group advances tensions from two or more past windows, flag all connections
and explain each separately. Converging past tensions in a single weak signal is itself
a sign of escalating consequence.

GROUPED WEAK SIGNAL — multiple current triples together constitute one weak signal.
If no individual current triple qualifies alone, but two or more read jointly develop a
past tension, add **one** `explanations` entry with `index` listing all bridging indices,
and a shared `theme` and `explanation`. Also include every index in `weak_signal_indices`.
For a single-triple weak signal, use a one-element list (e.g. `[2]`).

---

## Strict ignore list

Do NOT flag a current triple or group as a weak signal if:
- The only link to a past rare element is a shared domain label, shared actor name, or
  shared predicate type, with no shared underlying tension.
- The current triple(s) describe a development in the current baseline narrative that happens
  to mention the same topic as a past rare element.
- The connection requires inferring facts, causal mechanisms, or relationships not present
  in the provided triples.
- The flag depends on prior knowledge of the company, or of later events, that are not
  stated in the provided quintuples.
- Triples are grouped only because they share a domain or actor — grouping is only valid
  when the triples jointly constitute a step in a past tension that none captures alone.

---

## Structured output (required)

Return a single JSON object with exactly these fields:

### `weak_signal_indices` (list of integers)
Flat, deduplicated list of every flagged index into the **current bridging subgraph** [0..N-1].
Must equal the union of all `index` lists in `explanations`.

### `explanations` (list of objects)
One object per weak signal (single triple or group). Each object has:

- `index` (list of integers): one or more indices into the current bridging list. Use a list even for a single triple.
- `past_rare` (list of objects): the past rare triples whose tension this signal develops.
  Each object has:
  - `snapshot` (string): the past snapshot label as shown (e.g. `"2015"` from `--- Snapshot 2015 ---`).
  - `index` (list of integers): the displayed KG indices (bracket numbers) of the past **rare**
    triples that sourced the tension — not connective bridging edges that only link rares.
  Cite every past rare group whose tension this current signal develops. If multiple past
  snapshots contribute, include one `past_rare` entry per snapshot (or per rare group).
- `theme` (string): the shared underlying tension linking past rare patterns to this signal.
- `explanation` (string): how this triple or group develops that tension (mode of development, past source, concrete development). For groups, explain why the signal exists only when read together.

If no current triple or group qualifies, return empty lists and omit explanations.

---

## Calibration rules

- Current `index` values are 0-based into the CURRENT bridging subgraph.
- Past `past_rare.index` values are the displayed KG indices from the past blocks (not re-indexed).
- The baseline of each past snapshot matters: a triple is only a weak signal if its content
  was off-narrative then and the current triple (or group) continues that off-narrative thread.
- Prefer precision over recall. A plausible thematic connection is not sufficient.
  The tension must be traceable and the development must be concrete.
- Reason only from the numbered quintuples in this prompt. Treat the central entity as
  unknown besides those quintuples.
- Do not group triples merely to manufacture a signal.
- Stay grounded in the provided triples only. Do not invent facts.
"""

WEAK_SIGNAL_BRIDGING_USER = """Current snapshot label: {snapshot_label}

Past minimum bridging subgraphs (one block per prior snapshot):
{past_bridging_subgraphs}

Current snapshot minimum bridging subgraph (return indices into this list):
{current_bridging_triples}

Return structured output with:
- weak_signal_indices: flat deduplicated list of flagged indices into the current bridging list
- explanations: one item per weak signal; each has:
  - index: list of current bridging indices sharing the same theme and explanation
  - past_rare: list of {{snapshot, index}} citing the past rare KG indices whose tension this signal develops
  - theme and explanation"""

WEAK_SIGNAL_BRIDGING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", WEAK_SIGNAL_BRIDGING_SYSTEM),
    ("human", WEAK_SIGNAL_BRIDGING_USER),
])


def _render_prompt_template(prompt_template: ChatPromptTemplate, **variables: str) -> str:
    """Render a ChatPromptTemplate as plain text for LLM input or logging."""
    messages = prompt_template.format_messages(**variables)
    sections: list[str] = []
    for message in messages:
        role = message.type.upper()
        sections.append(f"=== {role} ===\n{message.content}")
    return "\n\n".join(sections)


def render_whole_snapshot_rare_prompt(**variables: str) -> str:
    """Render the whole-snapshot rare-detection prompt as plain text."""
    return _render_prompt_template(WHOLE_SNAPSHOT_RARE_DETECTION_PROMPT, **variables)


def render_weak_signal_bridging_prompt(**variables: str) -> str:
    """Render the weak-signal bridging prompt as plain text."""
    return _render_prompt_template(WEAK_SIGNAL_BRIDGING_PROMPT, **variables)
