"""Single source of truth for repository paths used by evaluation scripts."""
from pathlib import Path


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _find_repo_root()
DATASETS = REPO_ROOT / "datasets" / "c-unseen"
BENCHMARK = DATASETS / "openai_weak_signal_benchmark.json"
BENCHMARK_ANON = DATASETS / "openai_weak_signal_benchmark_anonymized.json"
SNAPSHOTS = DATASETS / "snapshots" / "openai_weak_signal"
CUNSEEN_PKG = REPO_ROOT / "itext2kg" / "c_unseen"
EVAL_ROOT = Path(__file__).resolve().parent
OUTPUTS = EVAL_ROOT / "outputs"
REPORTS = EVAL_ROOT / "reports"
SCORING = EVAL_ROOT / "scoring"
DETECTORS = EVAL_ROOT / "detectors"
MATCHING_SPEC = SCORING / "matching_spec.json"
ENTITY_MAP = SCORING / "entity_map.json"
OVERRIDES = SCORING / "overrides.json"
