"""How good is a concept-diagnosis vector? (DIAGNOSIS_CALIBRATION.md)

    python content/pkt/scripts/diagnosis_eval.py --split test \
        --cache last=mastery_cache.json ema=content/pkt/results/ksap_ema_inputs/mastery_cache.json \
                calibrated=content/pkt/results/ksap_calibrated/mastery_cache.json

Caches map "split|learner|cutoff" -> 37 scores (v2_cache_mastery.py). For every snapshot:
* item-level AUC / Brier: does the mean score of an attempt's concepts predict its verdict
  over the next 10 / 50 attempts (pools learners);
* within-learner Spearman: does the ranking of a learner's concepts match their actual success
  per concept over the next 50 attempts (concepts with >= 2 such attempts), averaged per
  learner; learner-bootstrap CIs for differences against the first cache.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score

PKT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKT))
from recommendation.data import iter_histories  # noqa: E402
from scripts.v2_cache_mastery import cutoffs_for  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=PKT.parents[1] / "data" / "codeforces.zip")
    p.add_argument("--split", choices=("valid", "test"), required=True)
    p.add_argument("--cache", nargs="+", required=True, help="name=path pairs; the first is the reference")
    args = p.parse_args()
    caches = {kv.split("=", 1)[0]: json.loads(Path(kv.split("=", 1)[1]).read_text()) for kv in args.cache}
    names = list(caches)
    hs = list(iter_histories(args.archive, args.split, include_asts=False))
    item = {(n, H): ([], []) for n in names for H in (10, 50)}
    within = {}
    for h in hs:
        per = {n: [] for n in names}
        for c in cutoffs_for(len(h.interactions)):
            key = f"{args.split}|{h.learner_id}|{c}"
            vec = {n: np.array(caches[n][key]) for n in names}
            for H in (10, 50):
                for e in h.interactions[c:c + H]:
                    idx = [t - 1 for t in e.concept_ids]
                    for n in names:
                        item[n, H][0].append(vec[n][idx].mean()); item[n, H][1].append(e.correct)
            fut = {}
            for e in h.interactions[c:c + 50]:
                for t in e.concept_ids:
                    fut.setdefault(t, []).append(e.correct)
            ks = [t for t, v in fut.items() if len(v) >= 2]
            y = np.array([np.mean(fut[t]) for t in ks])
            if len(ks) < 3 or y.std() == 0:
                continue
            for n in names:
                r = spearmanr(vec[n][[t - 1 for t in ks]], y).correlation
                if not np.isnan(r):
                    per[n].append(r)
        if all(per[n] for n in names):
            within[h.learner_id] = {n: float(np.mean(per[n])) for n in names}
    print(f"{args.split}: {len(hs)} learners; within-learner ranking over {len(within)} learners")
    print(f"{'diagnosis':12s} AUC@10  AUC@50  Brier@10  Brier@50  within-learner rho")
    for n in names:
        a10 = roc_auc_score(item[n, 10][1], item[n, 10][0]); a50 = roc_auc_score(item[n, 50][1], item[n, 50][0])
        b10 = brier_score_loss(item[n, 10][1], np.clip(item[n, 10][0], 0, 1))
        b50 = brier_score_loss(item[n, 50][1], np.clip(item[n, 50][0], 0, 1))
        rho = np.mean([within[i][n] for i in within])
        print(f"{n:12s} {a10:.3f}   {a50:.3f}   {b10:.3f}     {b50:.3f}     {rho:+.3f}")
    rng = np.random.default_rng(0)
    ids = list(within)
    for n in names[1:]:
        d = np.array([within[i][n] - within[i][names[0]] for i in ids])
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"within-learner rho, {n} - {names[0]}: {d.mean():+.3f}  95% CI "
              f"[{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]")


if __name__ == "__main__":
    main()
