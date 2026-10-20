# Calibrated concept diagnosis

16 October 2026. The goal was to improve the weakest component, the cognitive diagnosis.

## What it is

A small logistic layer on top of the frozen ASTNN-Longformer (`recommendation/calibration.py`). For each of the 37 concepts it combines:

| Feature | Source |
|---|---|
| model score at the last step, and its moving average (EMA) | frozen ASTNN-Longformer (code + history) |
| attempts on the concept, smoothed success rate on it, success on its last 5 attempts | learner's own record |
| overall and recent (last 20) success rate; attempts since the concept was last seen | learner's own record |

- **Fitting:** training learners only (16 learners, 24k rows), one row per (concept, future attempt). The target is solving the next 10 attempts that carry that concept, and each learner is weighted equally (`scripts/fit_calibration.py`).
- **Output:** a per-concept probability of success.
- **Run time:** numpy only. The weights are 8 coefficients in `results/ksap_calibrated/calibration.json`.

## Choice (validation learners) and confirmation (test, once)

Candidates were the current last-step score, the EMA, a counts-only model (no neural network), model outputs only, model + counts with logistic regression, and model + counts with gradient boosting. Model + counts with logistic regression was chosen on validation: it ranked a learner's own concepts best among the models that keep the neural network as an input. Counts-only was not significantly different on validation.

`scripts/diagnosis_eval.py` (files: `results/ksap_calibrated/diagnosis_{valid,test}.txt`):

| Test, 22 unseen learners | AUC@10 | AUC@50 | Brier@10 | Brier@50 | **within-learner ranking ρ** |
|---|---|---|---|---|---|
| Last step (current) | 0.550 | 0.570 | 0.322 | 0.311 | +0.107 |
| EMA | 0.565 | 0.584 | 0.279 | 0.275 | +0.144 |
| Counts only (no neural model) | 0.543 | 0.562 | 0.253 | 0.248 | +0.166 |
| **Calibrated (model + counts)** | 0.554 | 0.574 | **0.251** | **0.246** | **+0.209** |

How to read the columns:
- **Within-learner ranking ρ** is the property KSAP depends on. It asks whether the concepts ranked weakest for a learner are the ones they then fail most often (Spearman correlation per learner over the next 50 attempts, averaged over learners).
- **On ρ, the calibrated diagnosis roughly doubles the current one:** +0.101, learner-bootstrap 95% CI [+0.021, +0.178]. On validation the gain is +0.122 [+0.064, +0.185].
- **Calibration:** Brier 0.322 → 0.251. The raw head is overconfident (validation items scored 0.95 were solved only 66% of the time). Calibrated scores range from about 0.16 to 0.66 around the true 47% success rate.
- **AUC@10/50** pools learners together, and on it all estimators are close (0.55–0.61).
- **The neural model adds value:** on test, counts-only ranks concepts worse (+0.166) than the combination (+0.209).

## Behaviour with live input

Measured on the same 644 / 748 cut-offs as `MASTERY_SMOOTHING.md`: the change caused by one new attempt.

| | unrelated concepts | attempted concepts | right direction |
|---|---|---|---|
| Last step | 0.164 | 0.167 | 74% |
| EMA | 0.027 | 0.036 | 88% |
| Calibrated | 0.007 | 0.010 | 77% |

The calibrated estimate is the steadiest but reacts only slightly to a single submission. It also moves the wrong way more often than the EMA, because *having practised* a concept (even unsuccessfully) predicts later success on it. For live demos the EMA stays the default.

## KSAP on top of it (validation exploration only)

With calibrated scores, KSAP's fixed 0.5 threshold marks about 32 of 37 concepts as weak, so the learner-relative modes are the meaningful ones. On validation (`scripts/v2c_validation_grid.py`), every setting lies on one curve between relevance and remediation:

| Setting (calibrated diagnosis) | HR@10 | NDCG@10 | share on its 6 weakest |
|---|---|---|---|
| v2 (threshold 0.5) | 0.235 | 0.088 | 0.02 |
| 6 weakest, 3 per concept ("v2 targeted") | 0.129 | 0.041 | 0.82 |
| 10 weakest, 1 per concept (**demo: KSAP v2-C**) | 0.200 | 0.059 | 0.39 |
| 15 weakest, 1 per concept | 0.294 | 0.064 | 0.12 |
| 20 weakest, 2 per concept | 0.259 | 0.102 | 0.01 |

- The calibrated diagnosis ranks rarely practised topics (fft, 2-sat, flows, hashing) as weakest, because there is no evidence of mastery. Learners seldom attempt those topics next, so lists aimed at them score low on hit rate.
- No KSAP-on-calibrated configuration was run on test, and no recommendation-quality claim is made for it.
- **The reported KSAP results are unchanged** (last-step diagnosis, `KSAP_V2_RESULTS.md`).

## In the demo

- *Mastery estimate* has three options. Smoothed is the default. Choose Calibrated for the most accurate per-concept picture, and pair it with *Recommender → KSAP v2-C*.
- The weights are loaded from `results/ksap_calibrated/calibration.json`.

## Reproduce

```
python content/pkt/scripts/fit_calibration.py --archive data/codeforces.zip --checkpoint content/pkt/results/ksap_hf_v1/mastery.pt --output-dir content/pkt/results/ksap_calibrated
python content/pkt/scripts/v2_cache_mastery.py --archive data/codeforces.zip --checkpoint content/pkt/results/ksap_hf_v1/mastery.pt --output content/pkt/results/ksap_calibrated/mastery_cache.json --smoothing calibrated --calibration content/pkt/results/ksap_calibrated/calibration.json
python content/pkt/scripts/diagnosis_eval.py --split test --cache last=content/pkt/results/ksap_last_inputs/mastery_cache.json ema=content/pkt/results/ksap_ema_inputs/mastery_cache.json calibrated=content/pkt/results/ksap_calibrated/mastery_cache.json
```
