# Knowledge-tracing baselines: DKT, DKVMN, SAKT (report Section 5.4)

18 October 2026. Section 5.4 of the report plans to evaluate the system against DKT, DKVMN and SAKT. All three were implemented (`content/pkt/scripts/kt_baselines.py`) and compared with the ASTNN models in two roles:
1. as knowledge-tracing models (predicting the next attempt);
2. as the diagnosis engine inside KSAP (Objective 3).

## Setup
- **Inputs.** The same interaction stream as the ASTNN models' history (each attempt's concept tags and verdict), without the code.
- **Output.** The same as the ASTNN model: one logit per concept. A next-attempt prediction is the sigmoid of the mean logit over that attempt's tags. The vector after the latest attempt is the diagnosis that KSAP uses.
- **Data and training.** Codeforces split (16 train / 17 validation / 22 test learners). Adam, early stopping on validation AUC, **3 seeds** each.
- **Common protocol.** Every model is scored identically: test windows of up to 200 attempts, as in the original loader, and each prediction uses only earlier attempts in its window.
  - The ASTNN-LSTM baseline's original evaluation already satisfies this.
  - The ASTNN-Longformer checkpoint was scored with this protocol by `scripts/longformer_same_protocol.py`, using its original model code.

## 1. Next-attempt prediction (test, same 6,445 predictions, same protocol)

| Model | Uses code | Test AUC |
|---|---|---|
| DKT | no | **0.711 ± 0.001** |
| ASTNN-LSTM | yes | 0.704 |
| ASTNN-Longformer | yes | 0.636 |
| DKVMN | no | 0.622 ± 0.001 |
| SAKT | no | 0.618 ± 0.001 |

(± = standard deviation over 3 seeds; the ASTNN rows are single saved checkpoints.)

## 2. As the diagnosis engine of Objective 3

**Diagnosis quality** (`scripts/diagnosis_eval.py`, test; `results/kt_baselines/diagnosis_test.txt`):

| Diagnosis engine | AUC@10 | AUC@50 | Brier@50 | within-learner concept ranking ρ |
|---|---|---|---|---|
| ASTNN-Longformer (last step, as reported) | 0.550 | 0.570 | 0.311 | +0.107 |
| ASTNN-Longformer, **calibrated** (`DIAGNOSIS_CALIBRATION.md`) | 0.554 | 0.574 | 0.246 | **+0.209** |
| DKT | 0.546 | 0.566 | 0.260 | +0.210 |
| DKVMN | 0.574 | 0.584 | 0.246 | +0.230 |
| SAKT | 0.566 | 0.589 | 0.249 | +0.204 |

All three baselines rank a learner's own concepts better than the raw ASTNN-Longformer output:
- DKT +0.103 [+0.023, +0.185]
- DKVMN +0.123 [+0.050, +0.205]
- SAKT +0.096 [+0.004, +0.183]

The calibrated ASTNN diagnosis reaches the same level as the baselines (+0.209).

**Recommendation quality.** KSAP v2 was run with each engine; everything else is identical and frozen (`scripts/v2_final.py --set personalised`; `results/kt_baselines/ksap_*`):

| Diagnosis engine | KSAP v2 HR@10 | KSAP v2 NDCG@10 |
|---|---|---|
| ASTNN-Longformer (reported run) | 0.464 | 0.124 |
| DKT | 0.445 | 0.111 |
| DKVMN | 0.482 | 0.121 |
| SAKT | 0.518 | 0.137 |
| *Popularity (no diagnosis)* | *0.373* | *0.147* |
| *DMF only (no diagnosis)* | *0.427* | *0.131* |

Paired learner-bootstrap differences against the ASTNN-Longformer engine:
- DKT: HR −0.018 [−0.118, +0.073], NDCG −0.013 [−0.059, +0.027]
- DKVMN: HR +0.018 [−0.055, +0.118], NDCG −0.003 [−0.041, +0.034]
- SAKT: HR +0.055 [−0.018, +0.145], NDCG +0.012 [−0.028, +0.058]

**None of the differences is significant.** KSAP's recommendation quality does not depend measurably on which engine supplies the diagnosis.

(The "share on own 6 weakest" metric is relative to each engine's own diagnosis, so it is not compared across engines.)

## What this means for Objective 3
- **The framework is engine-agnostic.** Any knowledge-tracing model with a per-concept output plugs into the same KSAP pipeline; four engines were tested end to end.
- **At the concept level, the code-aware engine gives no measurable advantage for recommendation** on this data. Its raw concept ranking is weaker than the three baselines'. After the calibration layer, it matches them.
- What the ASTNN engine adds for the system is the **code channel**: pasted code is converted exactly into the trained format and influences the diagnosis (`OPTION_C_FEASIBILITY.md`, 5c). None of the baselines can use code at all.

## Reproduce
```
python content/pkt/scripts/kt_baselines.py --archive data/codeforces.zip --out content/pkt/results/kt_baselines
python content/pkt/scripts/longformer_same_protocol.py
python content/pkt/scripts/diagnosis_eval.py --split test --cache ASTNN-Longformer=content/pkt/results/ksap_last_inputs/mastery_cache.json DKT=content/pkt/results/kt_baselines/DKT_mastery_cache.json DKVMN=content/pkt/results/kt_baselines/DKVMN_mastery_cache.json SAKT=content/pkt/results/kt_baselines/SAKT_mastery_cache.json
python content/pkt/scripts/v2_final.py --archive data/codeforces.zip --bundle content/pkt/results/ksap_hf_v1 --training-mastery content/pkt/results/kt_baselines/DKT_full_mastery.json --full-mastery content/pkt/results/kt_baselines/DKT_full_mastery.json --cache content/pkt/results/kt_baselines/DKT_mastery_cache.json --split test --set personalised --output-dir <new dir>
```
