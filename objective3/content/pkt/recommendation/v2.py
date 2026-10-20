"""KSAP v2: Codeforces-specific refinements of the KSAP components.

Every refinement is an explicit, switchable option so that the original
friend implementation (v1) and each change can be compared in one harness:

* difficulty: Beta-smoothed empirical difficulty and equal-frequency
  (tertile) buckets instead of raw 1 - correct/attempts with fixed 0.3/0.7
  cut-offs, which are dominated by problems seen only once or twice.
* weak concepts: learner-relative selection (the k least-mastered concepts
  that still have catalogue candidates) instead of the fixed 0.5 threshold,
  which flags almost every concept because the head is uncalibrated.
* knowledge graph: Apriori baskets mined from fixed-size windows of each
  training learner's history instead of one basket per learner, which
  saturates (every learner eventually fails implementation/greedy/math...).
* ranking: DMF scores blended with a smoothed popularity prior so that the
  many exact DMF ties for items with identical training columns are broken
  by evidence rather than by problem ID.

All global artefacts are fitted only on the histories passed to ``fit`` (the
training learners for validation; training + validation for the final test).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from recommendation.data import LearnerHistory
from recommendation.dmf import load_dmf, train_dmf
from recommendation.knowledge_graph import fit_graph, weak_sequence
from recommendation.preferences import BUCKETS, difficulty_preference, fit_catalog
from recommendation.ranker import recommend as ksap_rank

NUM_CONCEPTS = 37


@dataclass(frozen=True)
class Config:
    name: str = "ksap_v1"
    # difficulty
    smoothing: float = 0.0            # Beta prior strength (pseudo-attempts); 0 = raw
    buckets: str = "paper"            # "paper" (0.3/0.7) or "quantile" (tertiles)
    use_dpm: bool = True              # False: no difficulty filter
    dpm_w1: float = 0.5               # DPM weight on attempt share (w2 = 1 - w1 on bucket accuracy)
    dpm_mode: str = "strict"          # "strict" (paper filter) or "soft" (preferred bucket ranked first)
    # weak concepts / ordering
    weak_mode: str = "threshold"      # "threshold" or "relative"
    weak_threshold: float = 0.5
    weak_k: int = 6
    weak_min_candidates: int = 0      # >0: a concept counts as weak only with >= this many unattempted problems
    prereq_rule: str = "all"          # "all": add every graph prerequisite (paper); "unmastered": only those
                                      # below the learner's own median mastery
    graph: str = "learner"            # "learner", "window", or "none" (order by mastery)
    window: int = 50
    # ranking
    ranker: str = "dmf"               # "dmf", "popularity", "hybrid"
    dmf: str = "v1"                   # which fitted DMF: "v1" (learner rows) or "win<N>" (window rows)
    dmf_recent: int = 0               # 0: score with the whole prefix row; N: only the last N attempts
    hybrid_weight: float = 0.5
    top_k_per_concept: int = 3
    allow_fallback: bool = False


def _windows(histories, size):
    out = []
    for h in histories:
        items = h.interactions
        for start in range(0, len(items), size):
            chunk = items[start:start + size]
            if len(chunk) >= max(5, size // 2):
                out.append(LearnerHistory(f"{h.learner_id}#{start}", tuple(chunk), h.source_split))
    return out


def build_catalog(histories, smoothing=0.0, buckets="paper"):
    catalog = fit_catalog(histories)
    total_attempts = sum(r["attempts"] for r in catalog.values())
    total_correct = sum(r["correct"] for r in catalog.values())
    prior = total_correct / total_attempts
    for row in catalog.values():
        if smoothing > 0:
            success = (row["correct"] + smoothing * prior) / (row["attempts"] + smoothing)
            row["difficulty"] = 1 - success
    if buckets == "quantile":
        values = np.array([r["difficulty"] for r in catalog.values()])
        lo, hi = np.quantile(values, [1 / 3, 2 / 3])
        for row in catalog.values():
            d = row["difficulty"]
            row["bucket"] = "easy" if d <= lo else ("moderate" if d < hi else "difficult")
        cut = (float(lo), float(hi))
    else:
        for row in catalog.values():
            d = row["difficulty"]
            row["bucket"] = "easy" if d <= 0.3 else ("moderate" if d < 0.7 else "difficult")
        cut = (0.3, 0.7)
    return catalog, {"prior_success": prior, "smoothing": smoothing, "bucket_cuts": cut}


@dataclass
class Artifacts:
    catalogs: dict = field(default_factory=dict)   # (smoothing, buckets) -> (catalog, meta)
    graphs: dict = field(default_factory=dict)     # mode -> graph
    dmf: object = None
    dmfs: dict = field(default_factory=dict)
    popularity: dict = field(default_factory=dict)
    concept_items: dict = field(default_factory=dict)
    mean_mastery: list = field(default_factory=list)
    fit_learners: list = field(default_factory=list)


def fit(histories, mean_mastery, dmf_checkpoint=None, dmf_epochs=30, seed=0, windows=(50,),
        smoothing_values=(0.0, 3.0), bucket_modes=("paper", "quantile"), dmf_windows=()):
    """Fit every switchable global artefact on ``histories`` only."""
    histories = list(histories)
    art = Artifacts(mean_mastery=list(mean_mastery), fit_learners=[h.learner_id for h in histories])
    for s in smoothing_values:
        for b in bucket_modes:
            art.catalogs[s, b] = build_catalog(histories, s, b)
    base = art.catalogs[smoothing_values[0], bucket_modes[0]][0]
    art.graphs["learner"] = fit_graph(histories, mean_mastery)
    for w in windows:
        art.graphs[f"window{w}"] = fit_graph(_windows(histories, w), mean_mastery)
    if dmf_checkpoint is None:
        dmf_checkpoint = train_dmf(histories, sorted(int(p) for p in base), epochs=dmf_epochs, seed=seed)
    art.dmf = load_dmf(dmf_checkpoint)
    art.dmfs["v1"] = art.dmf
    vocab = sorted(int(p) for p in base)
    for w in dmf_windows:
        art.dmfs[f"win{w}"] = load_dmf(train_dmf(_windows(histories, w), vocab, epochs=dmf_epochs, seed=seed))
    # Popularity: number of distinct fit learners who attempted the problem (smoothed log scale).
    learners_per_item = {}
    for h in histories:
        for pid in {e.problem_id for e in h.interactions}:
            learners_per_item[pid] = learners_per_item.get(pid, 0) + 1
    art.popularity = {int(p): math.log1p(learners_per_item.get(int(p), 0)) + 1e-3 * math.log1p(base[p]["attempts"])
                      for p in base}
    for p, row in base.items():
        for c in row["concept_ids"]:
            art.concept_items.setdefault(c, []).append(int(p))
    return art


def _zscore(values):
    arr = np.array(list(values.values()), dtype=float)
    mu, sd = arr.mean(), arr.std() or 1.0
    return {k: (v - mu) / sd for k, v in values.items()}


def recommend(art, cfg, prefix, mastery, limit=10):
    """Return KSAP-style recommendation dict for an observed prefix and its mastery vector."""
    catalog, _ = art.catalogs[cfg.smoothing, cfg.buckets]
    attempted = {e.problem_id for e in prefix}
    values = list(mastery)

    # 1. weak concepts
    remaining = {c: sum(i not in attempted for i in items) for c, items in art.concept_items.items()}
    has_candidates = {c for c, n_left in remaining.items() if n_left >= max(1, cfg.weak_min_candidates)}
    if cfg.weak_mode == "relative":
        ranked = sorted((c for c in range(1, NUM_CONCEPTS + 1) if c in has_candidates),
                        key=lambda c: (values[c - 1], c))
        weak = set(ranked[:cfg.weak_k])
        weak_threshold = max(values[c - 1] for c in weak) + 1e-9 if weak else 0.0
    else:
        weak_threshold = cfg.weak_threshold
        weak = {c for c in range(1, NUM_CONCEPTS + 1) if values[c - 1] < weak_threshold}
        if cfg.weak_min_candidates:
            weak &= has_candidates

    # 2. graph expansion and ordering
    if cfg.graph == "none":
        ordered = sorted(weak, key=lambda c: (values[c - 1], c))
        seq = {"weak_concepts": sorted(weak), "expanded_concepts": sorted(weak), "ordered_concepts": ordered}
    else:
        graph = art.graphs[cfg.graph if cfg.graph == "learner" else f"window{cfg.window}"]
        seq = weak_sequence(graph, values, threshold=min(1.0, weak_threshold),
                            weak=weak if cfg.weak_min_candidates else None,
                            prereq_below=float(np.median(values)) if cfg.prereq_rule == "unmastered" else None)
        if cfg.weak_mode == "relative":
            # expansion may add prerequisites; ordering stays topological
            pass

    # 3. explicit difficulty preference
    pref = difficulty_preference(prefix, catalog, w1=cfg.dpm_w1, w2=1 - cfg.dpm_w1)
    if not cfg.use_dpm:
        # A no-filter preference: give every item the same pseudo bucket.
        catalog = {k: {**v, "bucket": "moderate"} for k, v in catalog.items()}
        pref = {**pref, "selected_bucket": "moderate"}

    # 4. implicit preference scores
    ranker_model = art.dmfs.get(cfg.dmf, art.dmf)
    row_ids = attempted if not cfg.dmf_recent else {e.problem_id for e in prefix[-cfg.dmf_recent:]}
    dmf_scores = ranker_model.score(ranker_model.problem_ids, row_ids)
    if cfg.ranker == "dmf":
        scores = dmf_scores
    elif cfg.ranker == "popularity":
        scores = dict(art.popularity)
    else:
        zd, zp = _zscore(dmf_scores), _zscore(art.popularity)
        w = cfg.hybrid_weight
        scores = {p: (1 - w) * zd[p] + w * zp[p] for p in dmf_scores}

    if cfg.use_dpm and cfg.dpm_mode == "soft":
        preferred = pref["selected_bucket"]
        lo = min(scores.values())
        span = (max(scores.values()) - lo) or 1.0
        scores = {p: (v - lo) / span + (1.0 if catalog[str(p)]["bucket"] == preferred else 0.0)
                  for p, v in scores.items()}
        catalog = {k: {**v, "bucket": preferred} for k, v in catalog.items()}

    graph_for_warnings = art.graphs.get("learner") if cfg.graph == "none" else \
        art.graphs[cfg.graph if cfg.graph == "learner" else f"window{cfg.window}"]
    ranking = ksap_rank(catalog, seq["ordered_concepts"], pref, scores, attempted, values,
                        graph_for_warnings, cfg.top_k_per_concept, limit, cfg.allow_fallback)
    return {**seq, "difficulty_preference": pref, **ranking}


def expected_success(mastery, concept_ids):
    """Model-implied success chance for an item: mean of its concept probabilities."""
    return float(np.mean([mastery[c - 1] for c in concept_ids])) if concept_ids else float("nan")
