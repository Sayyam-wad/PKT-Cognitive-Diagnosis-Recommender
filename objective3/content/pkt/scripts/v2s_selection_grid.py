"""Validation-only grid that selected KSAP v2-S (MASTERY_SMOOTHING.md, section "End to end").

Uses the smoothed (EMA) mastery inputs in results/ksap_ema_inputs and the training-only DMF.
    python content/pkt/scripts/v2s_selection_grid.py
19 variants were compared (plus baselines). Pre-stated rule: restore weak-concept targeting to the
popularity level without lowering HR/NDCG below plain v2. No variant met it strictly; the chosen one
is not significantly worse on any of the three (learner bootstrap), which relaxed the rule after the fact.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

PKT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKT))
from recommendation import v2  # noqa: E402
from recommendation.data import iter_histories  # noqa: E402
from scripts.v2_experiments import evaluate  # noqa: E402

ARCHIVE = PKT.parents[1] / "data" / "codeforces.zip"
INPUTS = PKT / "results" / "ksap_ema_inputs"
BUNDLE = PKT / "results" / "ksap_hf_v1"


def main():
    train = list(iter_histories(ARCHIVE, "train", include_asts=False))
    valid = list(iter_histories(ARCHIVE, "valid", include_asts=False))
    tm = json.loads((INPUTS / "training_mastery.json").read_text())
    mean_m = [sum(v[i] for v in tm.values()) / len(tm) for i in range(37)]
    art = v2.fit(train, mean_m, dmf_checkpoint=str(BUNDLE / "dmf.pt"), windows=(100,))
    V1 = v2.Config(name="v1")
    V2 = replace(V1, name="v2", graph="window", window=100)
    grid = [V1, V2]
    for m in (3, 5, 10, 20):
        grid.append(replace(V2, name=f"v2 min{m}", weak_min_candidates=m))
    for m in (5, 10):
        grid.append(replace(V1, name=f"v1 min{m}", weak_min_candidates=m))
    for k in (4, 6, 8):
        for m in (1, 5):
            grid.append(replace(V2, name=f"v2 rel{k} min{m}", weak_mode="relative", weak_k=k, weak_min_candidates=m))
    grid.append(replace(V2, name="v2 nograph min5", graph="none", weak_min_candidates=5))
    for t in (1, 2):
        grid.append(replace(V2, name=f"v2 top{t}", top_k_per_concept=t))
        for k in (6, 8):
            grid.append(replace(V2, name=f"v2 rel{k} min5 top{t}", weak_mode="relative", weak_k=k,
                                weak_min_candidates=5, top_k_per_concept=t))
    res = evaluate(art, grid, valid, json.loads((INPUTS / "mastery_cache.json").read_text()), "valid")
    print(f"{'method':22s} HR     NDCG   Weak6  ExpS")
    for name, r in res["summary"].items():
        print(f"{name:22s} {r['hit_rate']:.3f}  {r['ndcg_at_k']:.3f}  {r['targets_6_weakest']:.2f}  {r['expected_success']:.3f}")


if __name__ == "__main__":
    main()
