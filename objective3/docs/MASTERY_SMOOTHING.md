# Mastery estimate: last step vs smoothed

16 October 2026. Found while testing live code input in the demo.

## Problem

The diagnosis vector is the frozen ASTNN-Longformer's concept prediction at the **last observed step**. The model was trained to predict the next attempt, so this vector is dominated by the most recent submission.

In the demo, adding one accepted solution to a *math, greedy* problem moved **sortings** from 0.05 to 0.995 and **suffix structures** from 0.07 to 0.91. Neither concept was involved. This is not specific to pasted code: it happens on the dataset's own submissions too (table below).

## What was tested

Each estimator was applied at 644 random cut-offs (validation learners) and 748 (test learners). Two prefixes were scored: one ending before an attempt, one including it.

- **Validity:** AUC for predicting the next attempt's verdict from the mean score of its concepts ("AUC next"), and for the attempt after that ("AUC +1").
- **Stability:** mean absolute change caused by one new attempt, on concepts *not* tagged on it ("off-tag"), on its own concepts ("on-tag"), and the largest off-tag change ("max off").
- **Direction:** share of attempts after which the attempted concepts moved the right way (up after a success, down after a failure).

Validation learners (used to choose):

| Estimator | AUC next | AUC +1 | off-tag \|Δ\| | on-tag \|Δ\| | max off \|Δ\| | direction ok |
|---|---|---|---|---|---|---|
| Last step (current) | 0.666 | 0.665 | 0.157 | 0.157 | 0.531 | 0.72 |
| Mean of last 5 | 0.666 | 0.669 | 0.059 | 0.081 | 0.175 | 0.87 |
| Mean of last 30 | 0.659 | 0.660 | 0.012 | 0.017 | 0.034 | 0.85 |
| EMA α = 0.3 | 0.665 | 0.666 | 0.058 | 0.076 | 0.173 | 0.89 |
| **EMA α = 0.1 (chosen)** | 0.661 | 0.666 | **0.026** | 0.036 | **0.073** | **0.89** |

Test learners (confirmation, not used to choose):

| Estimator | AUC next | AUC +1 | off-tag \|Δ\| | max off \|Δ\| | direction ok |
|---|---|---|---|---|---|
| Last step | 0.621 | 0.622 | 0.164 | 0.543 | 0.74 |
| EMA α = 0.1 | 0.600 | 0.620 | 0.027 | 0.075 | 0.88 |

- EMA minus last step, AUC next on test: −0.023, learner-bootstrap 95% CI [−0.048, +0.006]. **Not significant.**
- Stability improves about 6× and the direction is right far more often.
- Even with smoothing, on-tag changes are only slightly larger than off-tag (0.036 vs 0.026). The model shifts most concepts together, so it separates concepts only weakly. This matches the weak diagnosis already reported in `KSAP_V2_RESULTS.md`.

## Effect on KSAP (validation learners, everything else frozen)

| Method | HR@10 last → EMA | NDCG@10 last → EMA | share on 6 weakest last → EMA |
|---|---|---|---|
| KSAP v1 | 0.247 → 0.259 | 0.094 → 0.092 | 0.25 → 0.09 |
| KSAP v2 | 0.247 → 0.247 | 0.104 → 0.095 | 0.24 → 0.14 |
| Popularity (no diagnosis) | 0.200 → 0.200 | 0.072 → 0.072 | 0.30 → 0.31 |

Hit metrics are essentially unchanged. Weak-concept targeting drops: the smoothed estimate consistently ranks rare topics as weakest (2-sat: 14 catalogue problems; fft: 25; interactive: 72), and few problems exist to recommend for them.

## Decision

- **Reported evaluations keep the last-step estimate** (`HFMastery(smoothing="last")`, the default). The test set was evaluated once with it; switching after the fact would reuse the test set.
- **The demo offers both** ("Mastery estimate" in the sidebar), defaulting to the smoothed estimate. Live code input then gives believable updates; one pasted solution moved concepts by at most 0.11 in the example above, instead of 0.95.
- A smoothed end-to-end variant was then built and tested (next section). It is reported as a secondary analysis.

## End to end: smoothed diagnosis + KSAP v2-S (secondary analysis)

**Why targeting fell.** The cause turned out not to be rare concepts in the weak set: requiring at least *m* unattempted problems per weak concept (m = 3…20) changed nothing. The cause is the ordering. KSAP's knowledge graph puts broad prerequisites first (math, implementation, greedy), and three problems per concept fill the 10 slots before the weakest concepts are reached.

**Selection (validation learners only, smoothed mastery).**
- Grid: 19 variants of weak-set rule, minimum remaining problems, graph and problems per concept (`scripts/v2s_selection_grid.py`).
- Rule stated beforehand: bring weak-concept targeting up to the popularity level without lowering HR/NDCG below plain v2.
- No variant met it strictly. Every variant lies on one trade-off curve: ordering by weakness alone reached 100% targeting with HR 0.12, and the 4 weakest concepts reached 0.53 with HR 0.19.
- Chosen: **KSAP v2-S** = the 6 weakest concepts that still have at least 5 unattempted problems, graph order, **one problem per concept**. It is not significantly worse than v2 on NDCG (−0.007, CI [−0.023, +0.004]) or HR (+0.012, n.s.), and not significantly different from popularity on targeting (−0.010, n.s.).
- **The rule was relaxed to "not significantly worse" after seeing the grid.**

**Test (22 unseen learners, run once after freezing; `results/ksap_v2s_test`).**

| Method (smoothed diagnosis unless noted) | HR@10 | NDCG@10 | share on 6 weakest | concepts per list |
|---|---|---|---|---|
| KSAP v2, last-step diagnosis (main reported result) | 0.464 | 0.124 | 0.23 | — |
| KSAP v1 | 0.518 | 0.148 | 0.17 | 12.1 |
| KSAP v2 | 0.509 | 0.150 | 0.15 | 10.5 |
| **KSAP v2-S** | 0.464 | 0.103 | **0.33** | 15.8 |
| Popularity | 0.373 | 0.147 | 0.11 | 9.2 |
| DMF only | 0.427 | 0.131 | 0.14 | 11.3 |

Learner-bootstrap 95% CIs, v2-S minus:
- **KSAP v2:** targeting +0.18 [+0.12, +0.24]; NDCG −0.047 [−0.080, −0.017]; HR −0.045 (n.s.)
- **Popularity:** targeting +0.23 [+0.13, +0.32]; NDCG −0.044 (n.s.); HR +0.09 (n.s.)

**What this supports.**
1. **There is a real trade-off between relevance and remediation.** Learners mostly do not attempt their weakest topics next, so lists that focus on weaknesses score lower on "predict the next attempts" metrics. v2-S targets weak concepts significantly more, at a significant NDCG cost against v2. Which operating point is right depends on the goal, so it is a design choice, not a measured win.
2. **With the smoothed diagnosis, plain KSAP v2 scored higher on test than with the last-step diagnosis** (HR 0.509 vs 0.464, NDCG 0.150 vs 0.124), but targeted weak concepts less. This was not a pre-selected comparison (the smoothing was chosen for stability, not for these metrics), so treat it as an observation.
3. The main reported results (`KSAP_V2_RESULTS.md`, last-step diagnosis) are unchanged.

With one problem per concept, v2-S lists can be shorter than 10 when few concepts qualify (9.4 problems on average on test; other methods always return 10). This also lowers its precision.

In the demo, "KSAP v2-S" is available under *Recommender*. It pairs with the smoothed estimate.

## Evidence adjustment (live-input default)

The smoothed vector moves a submission's own concepts only slightly more than unrelated ones (0.036 vs 0.027), and in 12% of cases in the wrong direction. `calibration.evidence_adjust` therefore shifts each concept by the learner's record on it relative to their overall level:

`logit(out_c) = logit(ema_c) + β · [logit(rate_c) − logit(overall)]`

Here rate_c is shrunk towards the overall rate with k pseudo-attempts. Concepts never attempted are unchanged. β and k were chosen on validation (best within-learner ranking among settings with ≥ 95% right-direction updates): **β = 0.5, k = 2**.

| | within-learner ρ (valid / **test**) | AUC@50 (test) | right direction (test) | own vs unrelated change (test) |
|---|---|---|---|---|
| Smoothed | +0.191 / **+0.144** | 0.584 | 88% | 0.036 vs 0.027 |
| Smoothed + own record | +0.212 / **+0.116** | 0.584 | **94%** | **0.045 vs 0.026** |

- The ranking gain seen on validation did not hold on test: −0.028, CI [−0.071, +0.008], not significant. **No accuracy gain is claimed.**
- What held up: updates go the right way for a submission's own topics, and they move those topics more than unrelated ones.
- It is the demo's default for live input. The plain smoothed estimate remains selectable, and v2-S was evaluated with it.

## Reproduce

```
python content/pkt/scripts/mastery_smoothing_collect.py valid 40
python content/pkt/scripts/mastery_smoothing_analyze.py valid
python content/pkt/scripts/v2_cache_mastery.py --archive data/codeforces.zip --checkpoint content/pkt/results/ksap_hf_v1/mastery.pt --output mastery_cache_ema.json --smoothing ema
```

v2-S on test: `python content/pkt/scripts/v2_final.py --archive data/codeforces.zip --bundle content/pkt/results/ksap_hf_v1 --training-mastery content/pkt/results/ksap_ema_inputs/training_mastery.json --cache content/pkt/results/ksap_ema_inputs/mastery_cache.json --split test --set smoothed --full-mastery content/pkt/results/ksap_ema_inputs/valid_full_mastery.json --output-dir <new dir>`

The summary tables are in `content/pkt/results/mastery_smoothing_{valid,test}.txt` and `content/pkt/results/ksap_v2_valid_ema_summary.csv`. (The KSAP run also needs training-learner mastery recomputed with EMA. `v2_final.py` reads it from the bundle's `training_mastery.json`.)
