"""Frozen KSAP v1/v2 comparison with ablations on validation and test.

Configuration was selected on validation only (see V2_SELECTION below).
* validation run: global artefacts fitted on the 16 training learners, using
  the friend's original DMF checkpoint (identical to the v1 bundle).
* test run (executed once, after freezing): artefacts refitted on training +
  validation learners (33), DMF retrained with the same settings and seed;
  evaluated on the 22 unseen test learners.
"""

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

PKT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKT_ROOT))
from recommendation import v2  # noqa: E402
from recommendation.data import iter_histories  # noqa: E402
from scripts.v2_experiments import bootstrap_ci, evaluate, print_table  # noqa: E402

V1 = v2.Config(name="KSAP v1 (paper-faithful)")
V2 = replace(V1, name="KSAP v2", graph="window", window=100)
CONFIGS = [
    V1,
    V2,
    replace(V2, name="v2 - knowledge graph (order by mastery)", graph="none"),
    replace(V2, name="v2 - DPM (no difficulty filter)", use_dpm=False),
    replace(V2, name="v2 - DMF (popularity ranking)", ranker="popularity"),
    replace(V2, name="v2 - diagnosis (all concepts weak)", weak_threshold=1.0),
    replace(V2, name="v2 targeted (6 weakest concepts)", weak_mode="relative", weak_k=6),
]
V2_SELECTION = ("Selected on validation from 60+ variants: window-basket graph (W=100) was the only change that "
                "kept all four KSAP components and improved NDCG@10 (0.094 -> 0.104) without lowering HR@10.")
# Secondary analysis (MASTERY_SMOOTHING.md): smoothed (EMA) mastery end to end. Selected on validation
# with EMA mastery from a 19-variant grid (scripts/v2s_selection_grid.py): 6 weakest concepts that still have >= 5 unattempted problems,
# one problem per concept. Run with --set smoothed and an EMA mastery cache / bundle.
V2S = replace(V2, name="KSAP v2-S (smoothed diagnosis)", weak_mode="relative", weak_k=6,
              weak_min_candidates=5, top_k_per_concept=1)
# Personalised prerequisites (selected on validation, 18 Oct): the knowledge graph adds a prerequisite only
# if the learner's mastery of it is below their own median, so learners who already know the basics are not
# all sent to the same math/implementation/greedy problems first.
V2P = replace(V2, name="KSAP v2-P (personalised prerequisites)", prereq_rule="unmastered")
SETS = {"main": CONFIGS, "smoothed": [V1, V2, V2S], "personalised": [V1, V2, V2P]}
BASELINE_NAMES = {"random": "Random", "popularity": "Popularity", "dmf_only": "DMF only (no KSAP filters)"}


def mean_mastery_of(model_cache, histories):
    vectors = [model_cache[h.learner_id] for h in histories]
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(37)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", choices=("valid", "test"), required=True)
    p.add_argument("--full-mastery", type=Path, help="JSON learner->full-history mastery for validation learners (test fit)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--training-mastery", type=Path,
                   help="override the bundle's training_mastery.json (e.g. results/ksap_ema_inputs/training_mastery.json)")
    p.add_argument("--set", choices=tuple(SETS), default="main",
                   help="'main' = reported results; 'smoothed' = secondary EMA analysis (needs EMA cache and bundle)")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    train = list(iter_histories(args.archive, "train", include_asts=False))
    held = list(iter_histories(args.archive, args.split, include_asts=False))
    full = json.loads((args.training_mastery or args.bundle / "training_mastery.json").read_text())
    if args.split == "valid":
        fit_h, dmf_ckpt = train, str(args.bundle / "dmf.pt")
    else:
        valid = list(iter_histories(args.archive, "valid", include_asts=False))
        full.update(json.loads(args.full_mastery.read_text()))
        fit_h, dmf_ckpt = train + valid, None
    assert not {h.learner_id for h in fit_h} & {h.learner_id for h in held}
    art = v2.fit(fit_h, mean_mastery_of(full, fit_h), dmf_checkpoint=dmf_ckpt, windows=(100,))
    cache = json.loads(args.cache.read_text())
    configs = SETS[args.set]
    focus = {"main": V2, "smoothed": V2S, "personalised": V2P}[args.set]
    res = evaluate(art, configs, held, cache, args.split)

    # rename baselines for presentation
    for d in [res["summary"]] + list(res["per_learner"].values()):
        for old, new in BASELINE_NAMES.items():
            d[new] = d.pop(old)
    order = ["Random", "Popularity", "DMF only (no KSAP filters)"] + [c.name for c in configs]
    print_table(res, order)
    cis = {}
    pairs = ((V2.name, V1.name),) if args.set == "main" else ((focus.name, V2.name), (focus.name, V1.name))
    for a, b in pairs + ((focus.name, "Popularity"), (focus.name, "DMF only (no KSAP filters)")):
        for key in ("ndcg_at_k", "hit_rate", "targets_6_weakest"):
            cis[f"{a} vs {b} | {key}"] = bootstrap_ci(res["per_learner"], a, b, key)
    for k, v in cis.items():
        print(f"{k}: diff={v[0]:+.3f}  95% CI [{v[1]:+.3f}, {v[2]:+.3f}]")
    res["bootstrap_learner_ci"] = cis
    res["configs"] = [asdict(c) for c in configs]
    res["selection_note"] = V2_SELECTION
    res["fit_learners"] = len(fit_h)
    res["graph_stats"] = {k: g["stats"] for k, g in art.graphs.items()}
    (args.output_dir / "results.json").write_text(json.dumps(res, indent=1))
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        keys = ["hit_rate", "precision_at_k", "recall_at_k", "ndcg_at_k", "expected_success", "targets_6_weakest", "distinct_concepts"]
        w.writerow(["method", "HR@10", "Precision@10", "Recall@10", "NDCG@10", "model_expected_success",
                    "share_targeting_6_weakest", "distinct_concepts_in_list"])
        for n in order:
            w.writerow([n] + [round(res["summary"][n][k], 4) for k in keys])
    with (args.output_dir / "per_learner.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["learner", "method", "HR@10", "NDCG@10", "targets_6_weakest"])
        for lid, d in res["per_learner"].items():
            for n in order:
                w.writerow([lid, n, round(d[n]["hit_rate"], 4), round(d[n]["ndcg_at_k"], 4), round(d[n]["targets_6_weakest"], 4)])


if __name__ == "__main__":
    main()
