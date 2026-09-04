"""
Apply the entity anonymization map to the Wiki-OpenAI benchmark.

Usage:
    python anonymize.py entity_map.json benchmark.json out_anon.json

Produces:
    out_anon.json           anonymized benchmark
    out_anon.audit.json     coverage report: unmapped capitalised tokens,
                            placeholder collisions, per-field replacement counts

The map is applied case-sensitively, longest-surface-form-first, with word
boundaries. Timestamps, roles and precursor links are never modified.
"""

import json
import re
import sys
from collections import Counter, defaultdict

TEXT_FIELDS = ("text", "subject", "object", "off_narrative_rationale",
               "name", "description")

# Identifier fields replaced by EXACT lookup against the signal_ids map,
# not by regex. These are ground-truth annotation fields (never prompted to
# the model), but renaming them keeps the released artefact self-consistent.
ID_FIELDS = ("id", "arc", "signal")

# Tokens that legitimately start a sentence or are domain-general; excluded
# from the "unmapped capitalised token" audit to keep the report readable.
AUDIT_STOPWORDS = {
    "A", "An", "The", "In", "On", "At", "By", "For", "From", "To", "With",
    "After", "Before", "During", "Through", "Throughout", "Following",
    "This", "That", "These", "Those", "It", "Its", "As", "Of", "And", "Or",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "AI", "AGI", "API", "APIs", "LLM", "LLMs", "CEO", "CTO", "COO", "CFO",
    "Board", "Directors", "Chief", "Scientist", "President", "Chairman",
    "Officer", "Research", "Technology", "Company", "Corporation", "Inc",
    "LLC", "LP", "PBC", "Nonprofit", "Foundation",
}


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_rules(emap):
    """Flatten every category into an ordered (pattern, replacement) list."""
    skip = {"_meta", "replacement_rules", "residual_leakage_notes",
            "scope_decision"}
    pairs, sources = [], {}
    for category, entries in emap.items():
        if category in skip or not isinstance(entries, dict):
            continue
        for surface, placeholder in entries.items():
            if not isinstance(placeholder, str):
                continue
            pairs.append((surface, placeholder))
            sources.setdefault(placeholder, set()).add(category)

    # Longest first so "OpenAI Global, LLC" is consumed before "OpenAI".
    pairs.sort(key=lambda kv: len(kv[0]), reverse=True)

    rules = []
    for surface, placeholder in pairs:
        esc = re.escape(surface)
        # \b fails next to non-word chars (e.g. "Q*", "U.S."); guard instead.
        left = r"(?<![\w])" if surface[0].isalnum() else r""
        right = r"(?![\w])" if surface[-1].isalnum() else r""
        rules.append((re.compile(left + esc + right), placeholder, surface))
    return rules, sources


def check_collisions(emap, sources):
    """A placeholder reused across categories is almost always a mistake."""
    problems = []
    for placeholder, cats in sources.items():
        if len(cats) > 1:
            problems.append({"placeholder": placeholder,
                             "categories": sorted(cats)})
    return problems


def apply_rules(text, rules, counter):
    if not isinstance(text, str):
        return text
    for pattern, placeholder, surface in rules:
        text, n = pattern.subn(placeholder, text)
        if n:
            counter[surface] += n
    return text


def walk(node, rules, counter, id_map):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in TEXT_FIELDS:
                out[k] = apply_rules(v, rules, counter)
            elif k in ID_FIELDS and isinstance(v, str) and v in id_map:
                out[k] = id_map[v]
                counter[v] += 1
            else:
                out[k] = walk(v, rules, counter, id_map)
        return out
    if isinstance(node, list):
        return [walk(x, rules, counter, id_map) for x in node]
    return node


def audit_residual(node, found):
    """Collect capitalised tokens that survived replacement."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in TEXT_FIELDS and isinstance(v, str):
                for tok in re.findall(r"\b[A-Z][A-Za-z0-9.\-']{1,}\b", v):
                    if tok not in AUDIT_STOPWORDS and not tok.startswith(
                            ("Org", "Person", "Model", "Prod", "Gov",
                             "Media", "Univ", "Loc", "Doc", "Event",
                             "SS_", "ARC_", "Topic")):
                        found[tok] += 1
            else:
                audit_residual(v, found)
    elif isinstance(node, list):
        for x in node:
            audit_residual(x, found)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    map_path, bench_path, out_path = sys.argv[1:4]

    emap = load(map_path)
    bench = load(bench_path)

    rules, sources = build_rules(emap)
    collisions = check_collisions(emap, sources)

    counter = Counter()
    id_map = emap.get("signal_ids", {})
    anon = walk(bench, rules, counter, id_map)

    residual = Counter()
    audit_residual(anon, residual)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(anon, fh, indent=2, ensure_ascii=False)

    audit = {
        "rules_loaded": len(rules),
        "total_replacements": sum(counter.values()),
        "distinct_surfaces_matched": len(counter),
        "surfaces_never_matched": sorted(
            s for _, _, s in rules if counter[s] == 0),
        "replacements_by_surface": dict(counter.most_common()),
        "placeholder_collisions": collisions,
        "residual_capitalised_tokens": dict(residual.most_common(80)),
    }
    audit_path = out_path.replace(".json", "") + ".audit.json"
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    print(f"rules loaded            : {audit['rules_loaded']}")
    print(f"total replacements      : {audit['total_replacements']}")
    print(f"surfaces never matched  : {len(audit['surfaces_never_matched'])}")
    print(f"placeholder collisions  : {len(collisions)}")
    print(f"residual capitalised    : {len(residual)} distinct tokens")
    print(f"\nwrote {out_path}\nwrote {audit_path}")
    if residual:
        print("\nTop residual tokens (review these — each may be a leak):")
        for tok, n in residual.most_common(25):
            print(f"  {tok:<40} {n}")


if __name__ == "__main__":
    main()