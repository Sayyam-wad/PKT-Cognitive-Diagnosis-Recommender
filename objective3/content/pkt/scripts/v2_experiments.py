"""Multi-cutoff offline evaluation of KSAP v1, v2 variants and baselines.

Protocol: for each held-out learner and each cutoff in v2_cache_mastery
(50..90% of record order) the recommender sees only the prefix. Relevant items
are problems attempted after the cutoff that were not attempted before it.
Metrics are averaged over a learner's cutoffs first, then macro-averaged over
learners, so long histories do not dominate.
"""

import argparse
import json
from pathlib import Path
import random
import sys
from statistics import mean

import numpy as np

PKT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKT_ROOT))
from recommendation import v2  # noqa: E402
from recommendation.data import iter_histories  # noqa: E402
from recommendation.metrics import ranking_metrics  # noqa: E402
from scripts.v2_cache_mastery import cutoffs_for  # noqa: E402

METRICS = ("hit_rate", "precision_at_k", "recall_at_k", "ndcg_at_k")


def baseline_lists(art, prefix, k, rng):
    attempted = {e.problem_id for e in prefix}
    cands = [p for p in art.dmf.problem_ids if p not in attempted]
    dmf = art.dmf.score(art.dmf.problem_ids, attempted)
    return {
        "random": rng.sample(cands, k),
        "popularity": sorted(cands, key=lambda p: (-art.popularity[p], p))[:k],
        "dmf_only": sorted(cands, key=lambda p: (-dmf[p], p))[:k],
    }


def evaluate(art, configs, histories, cache, split, k=10, seed=0, cutoff_fn=cutoffs_for):
    rng = random.Random(seed)
    per_learner = {}
    snapshots = []
    for h in histories:
        n = len(h.interactions)
        if n < 2:
            continue
        rows = {}
        for c in cutoff_fn(n):
            prefix = h.interactions[:c]
            mastery = cache[f"{split}|{h.learner_id}|{c}"]
            attempted = {e.problem_id for e in prefix}
            lists, details = baseline_lists(art, prefix, k, rng), {}
            for cfg in configs:
                out = v2.recommend(art, cfg, prefix, mastery, limit=k)
                lists[cfg.name] = [r["problem_id"] for r in out["recommendations"]]
                details[cfg.name] = out
            relevant = {e.problem_id for e in h.interactions[c:]} - attempted
            if not relevant:
                continue
            weak_rel = set(sorted(range(1, 38), key=lambda i: (mastery[i - 1], i))[:6])
            snap = {"learner_id": h.learner_id, "cutoff": c, "n": n, "relevant": len(relevant), "methods": {}}
            for name, ids in lists.items():
                assert len(ids) == len(set(ids)) and not set(ids) & attempted, name
                m = ranking_metrics(ids, relevant, k)
                cat = art.catalogs[0.0, "paper"][0]
                es = [v2.expected_success(mastery, cat[str(p)]["concept_ids"]) for p in ids]
                weak_hit = [bool(set(cat[str(p)]["concept_ids"]) & weak_rel) for p in ids]
                m["returned"] = len(ids)
                m["expected_success"] = float(np.mean(es)) if es else None
                m["targets_6_weakest"] = float(np.mean(weak_hit)) if ids else None
                m["distinct_concepts"] = len({c for p in ids for c in cat[str(p)]["concept_ids"]})
                snap["methods"][name] = m
                rows.setdefault(name, []).append(m)
            snapshots.append(snap)
        if rows:
            per_learner[h.learner_id] = {
                name: {key: mean([r[key] for r in rs if r[key] is not None]) if any(r[key] is not None for r in rs) else None
                       for key in rs[0]} for name, rs in rows.items()}
    names = list(next(iter(per_learner.values())).keys())
    summary = {}
    for name in names:
        summary[name] = {}
        for key in per_learner[next(iter(per_learner))][name]:
            vals = [pl[name][key] for pl in per_learner.values() if pl[name][key] is not None]
            summary[name][key] = mean(vals) if vals else None
    return {"split": split, "k": k, "learners": len(per_learner), "snapshots": len(snapshots),
            "summary": summary, "per_learner": per_learner, "snapshot_rows": snapshots}


def bootstrap_ci(per_learner, a, b, key="ndcg_at_k", reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    ids = [i for i in per_learner if per_learner[i][a][key] is not None and per_learner[i][b][key] is not None]
    diffs = np.array([per_learner[i][a][key] - per_learner[i][b][key] for i in ids])
    boots = [diffs[rng.integers(0, len(ids), len(ids))].mean() for _ in range(reps)]
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def print_table(result, names=None):
    s = result["summary"]
    names = names or list(s)
    print(f"{'method':34s} HR@10  P@10   R@10   NDCG@10 ExpSucc Weak6 Concepts")
    for n in names:
        r = s[n]
        print(f"{n:34s} {r['hit_rate']:.3f}  {r['precision_at_k']:.3f}  {r['recall_at_k']:.3f}  {r['ndcg_at_k']:.3f}   "
              f"{r['expected_success']:.3f}   {r['targets_6_weakest']:.2f}  {r['distinct_concepts']:.1f}")
