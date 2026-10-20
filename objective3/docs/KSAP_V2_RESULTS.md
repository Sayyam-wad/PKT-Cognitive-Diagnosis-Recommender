# Objective 3: KSAP recommender, v2 evaluation and demo

Prepared 16 October 2026. This document covers the KSAP recommender built on the first (v1) implementation: a stronger evaluation protocol, KSAP v2 adaptations, an ablation study, a check on the cognitive diagnosis, learning-material links, and an interactive demo.

## 1. What the system does

For a learner's observed Codeforces attempts:

1. **Cognitive diagnosis.** The frozen ASTNN–Longformer checkpoint (Hugging Face `Sayyam9/astnn_longformer`, SHA-256 `b4be633b…96a4`, identical to `results_final/best_longformer_model.pt`) scores the 37 concepts from the observed prefix only.
2. **Knowledge structure (KSAP §4.2).** Weak concepts (score < 0.5) are expanded with association-rule prerequisites and ordered topologically on a concept graph mined with Apriori.
3. **Attempting preferences (KSAP §4.3).** DPM picks the learner's explicit difficulty bucket. DMF scores implicit preference from learners with similar attempt histories.
4. **Recommendation (KSAP Algorithm 3).** For each ordered concept, unseen problems of the preferred difficulty are ranked by DMF, three per concept, ten in total.
5. **Learning material (new).** Each recommendation names its target concept, explains why it was chosen, and links reading material (CP-Algorithms / Codeforces EDU) plus a Codeforces practice filter for that concept (`recommendation/resources.py`). Concept names come from `data/tag_table.csv` of the source repo `pkt-attn/pkt-attn`.

## 2. Evaluation protocol (changed from the handoff)

- **Learners.** Global components (catalogue, graph, DMF) are fitted on training learners only. For validation: 16 training learners. For the single final test run: training plus validation, 33 learners, with DMF retrained using the same settings and seed. Evaluated learners are never used for fitting.
- **Snapshots.** Each held-out learner is evaluated at 5 cut-offs (50, 60, 70, 80 and 90% of their record) instead of one: 85 validation snapshots and 108 test snapshots. Only the prefix is visible. Relevant items are problems attempted after the cut-off that were not attempted before it.
- **Aggregation.** Metrics are averaged over a learner's snapshots first, then over learners, so long histories don't dominate. 95% confidence intervals use a bootstrap over learners.
- **Metrics:**
  - Agreement with future attempts: HR@10, Precision@10, Recall@10, NDCG@10.
  - **Targeting:** the share of recommended problems tagged with one of the learner's 6 weakest diagnosed concepts.
  - **Expected success:** the model's mean concept score for the recommended problems.
  - **Coverage:** distinct concepts in the list.
- **Cross-check.** At the 80% cut-off the harness reproduces the handoff's v1 and DMF-only numbers exactly (HR 0.1765, NDCG 0.0714 / 0.0951). Popularity is defined here as the number of distinct training learners who attempted a problem, not raw attempt counts.

## 3. What was changed and why (KSAP v2)

Problems found in v1 and in the earlier local pipeline (`models/astnn/build_*.py`):

| Problem | Evidence | v2 response |
|---|---|---|
| One incorrect-concept basket per learner saturates. Almost every learner eventually fails implementation, greedy, math, dp and strings, so every rule points into those concepts. | Local `ksap_edges.csv`: all 99 edges target {1, 3, 6, 7, 8, 15} | Mine baskets from **100-attempt windows** (`graph="window", window=100`) |
| The uncalibrated head flags most concepts as weak at 0.5 | Ali.Kh @100: 29/37 weak | Tested a learner-relative weak set (6 weakest) as "v2 targeted" |
| Raw difficulty is noisy: 1,030 of 3,990 problems have a single training attempt | catalogue statistics | Tested Beta-smoothed difficulty + tertile buckets |
| DPM almost always selects "easy": success rate is always highest on easy problems | 13–16 of 17 validation learners | Tested DPM weights w1 ∈ {0.5, 0.75, 1.0} (the paper leaves them unspecified) |
| DMF ties for items with identical training columns | identical scores at top ranks | Tested popularity blend and window-trained DMF |

More than 60 variants were compared on **validation only**. The only change that kept all four KSAP components and improved validation NDCG@10 without lowering HR@10 was the window-basket graph, so **KSAP v2 = v1 + window graph (W = 100)**. Smoothing, relative weak sets, DPM re-weighting, popularity blending and window-trained DMF did not improve validation agreement and were not adopted. They remain available as `Config` switches.

## 4. Results

### Test (22 unseen learners, 108 snapshots; run once after freezing)

| Method | HR@10 | P@10 | R@10 | NDCG@10 | Targets 6 weakest | Concepts in list |
|---|---|---|---|---|---|---|
| Random | 0.120 | 0.013 | 0.013 | 0.014 | 0.25 | 14.5 |
| Popularity | 0.373 | 0.144 | 0.044 | 0.147 | 0.12 | 9.2 |
| DMF only (no KSAP filters) | 0.427 | 0.136 | 0.024 | 0.131 | 0.21 | 11.3 |
| **KSAP v1 (paper-faithful)** | **0.473** | 0.118 | 0.021 | 0.120 | **0.32** | 13.9 |
| **KSAP v2** | 0.464 | 0.125 | 0.025 | 0.124 | 0.23 | 10.4 |
| KSAP v2 targeted (6 weakest) | 0.473 | 0.113 | 0.024 | 0.111 | 0.30 | 11.5 |

Ablation on test, removing one component from v2 at a time:

| Variant | HR@10 | NDCG@10 | Targets 6 weakest |
|---|---|---|---|
| KSAP v2 | 0.464 | 0.124 | 0.23 |
| − knowledge graph (weakest-first, no expansion) | 0.273 | 0.048 | 1.00 |
| − DPM (no difficulty filter) | 0.400 | 0.125 | 0.30 |
| − DMF (popularity ranking) | 0.409 | 0.105 | 0.22 |
| − diagnosis (all concepts treated as weak) | 0.527 | 0.141 | 0.15 |

Learner-bootstrap 95% CIs on test:

| Comparison | Metric | Difference | 95% CI |
|---|---|---|---|
| v2 vs v1 | NDCG | +0.005 | [−0.015, +0.022] |
| v2 vs v1 | targeting | −0.085 | [−0.115, −0.056] |
| v2 vs Popularity | NDCG | −0.022 | [−0.094, +0.038] |
| v2 vs Popularity | HR | +0.091 | [−0.091, +0.255] |
| v2 vs Popularity | targeting | **+0.113** | **[+0.058, +0.171]** |
| v2 vs DMF only | NDCG | −0.007 | [−0.042, +0.029] |

### Validation (17 learners, 85 snapshots)

KSAP v1 has HR 0.247 / NDCG 0.094, KSAP v2 has 0.247 / 0.104, DMF only has 0.294 / 0.117, and popularity has 0.200 / 0.072. Full tables are in `results/ksap_v2_valid/summary.csv`.

### How to read these results (claims the data supports)

1. **KSAP recommendations agree with what learners actually attempted next at least as well as data-driven baselines.** On test, KSAP v1 and v2 have the highest HR@10 (0.47 vs 0.43 DMF-only and 0.37 popularity). NDCG differences are within noise.
2. **KSAP is significantly more targeted.** Its lists aim at the learner's diagnosed weak concepts more than popularity's do (+0.11, CI excludes 0), and every item comes with an explanation and learning material.
3. **There is a trade-off between agreement and targeting.** Recommending strictly the weakest concepts (no graph) gives 100% targeting but much lower agreement. The knowledge graph's prerequisite expansion is what keeps KSAP's lists plausible: it routes learners through prerequisite concepts first (see `figures/test_tradeoff.png`).
4. **v2's graph change did not generalise.** It helped on validation, but on test v1 and v2 are statistically tied, and v1 targets weak concepts more. Report this honestly. With 17–22 learners per split, differences of a few hundredths are within noise.
5. **The diagnosis is the weakest link.** Removing it (treating every concept as weak) gives the highest agreement. See section 5.

## 5. How informative is the cognitive diagnosis?

`diag` check: score each held-out snapshot, then ask whether the concept scores predict correctness on the learner's next 10/50 attempts.

| Concept-score source | Valid next 10 | Valid next 50 | Test next 10 | Test next 50 |
|---|---|---|---|---|
| HF ASTNN–Longformer (used) | 0.582 | 0.568 | 0.550 | 0.570 |
| Longformer retrained with prefix-only attention | 0.595 | 0.600 | 0.541 | 0.558 |
| Per-concept success-rate count | 0.594 | 0.584 | 0.539 | 0.550 |

The values are AUC. The scores carry real but weak signal, comparable to the alternatives, and they are overconfident. On validation, items scored 0.19 were solved 42% of the time, and items scored 0.95 were solved 66% of the time. Treat the concept scores as rankings within a learner rather than as calibrated probabilities. Calibrating the head, or training a dedicated concept head with a causal mask, is the clearest next improvement.

**Stability (added 16 Oct).** The concept vector is the model's prediction at the last observed step, and it swings with each new submission. One attempt moves *unrelated* concepts by 0.16 on average and by up to 0.54. A moving average of the per-step predictions (EMA, α = 0.1, chosen on validation) cuts that to 0.03 with no significant loss of next-attempt AUC. The trade-off is that KSAP's weak-concept targeting drops on validation, so the results above keep the last-step estimate. The demo offers both. Details: `MASTERY_SMOOTHING.md`. A secondary end-to-end variant (smoothed diagnosis + KSAP v2-S) targets weak concepts significantly more on test (0.33 vs 0.15 for v2 on the same diagnosis), at a significant NDCG cost. It is reported there as a relevance-vs-remediation trade-off.

**Calibrated diagnosis (added 16 Oct).** A logistic layer over the model's outputs and the learner's record, fitted on training learners, ranks a learner's own concepts about twice as well as the raw scores. On test the within-learner Spearman is 0.209 vs 0.107 (difference +0.101, CI [+0.021, +0.178]), and the scores are calibrated (Brier 0.251 vs 0.322). The neural model contributes beyond the counts: counts alone reach 0.166. KSAP on top of it was only explored on validation. Details: `DIAGNOSIS_CALIBRATION.md`.

## 5b. Personalisation: why lists looked alike, and KSAP v2-P (added 18 Oct)

**Finding.** KSAP's knowledge graph adds every *prerequisite* of a weak concept, whether or not the learner already knows it, and puts prerequisites first in the order. Nearly all weak concepts lead back to the same basics (math → implementation → constructive algorithms → greedy). As a result, 21 of 22 test learners received the same first three target topics. The lists still differ (no two are identical; average overlap 12%), but the top of the list is shared.

**Change (option `prereq_rule="unmastered"`, chosen on validation).** A prerequisite is added only if the learner's mastery of it is below their own median. A learner who already knows implementation is not sent back to implementation problems.

| Test, 22 unseen learners (last-step diagnosis, reported protocol) | HR@10 | NDCG@10 | share on own 6 weakest | same first-3 topics |
|---|---|---|---|---|
| KSAP v2 | 0.464 | 0.124 | 0.23 | 21 / 22 learners |
| **KSAP v2-P** | 0.382 | 0.086 | **0.45** | **3 / 22** |

Learner-bootstrap 95% CIs, v2-P minus v2:
- targeting **+0.22 [+0.16, +0.28]** (significant);
- HR −0.08 [−0.18, +0.02] (not significant);
- NDCG −0.04 [−0.08, +0.00] (not significant).

The lists become individual and remediation-focused, at a cost in "predict the next attempts" metrics that is not significant. This is the same relevance-vs-remediation trade-off as in `MASTERY_SMOOTHING.md`. The main results above are unchanged. In the demo, choose *Recommender → KSAP v2-P* to show personalised lists. Files: `results/ksap_v2p_test`; `scripts/v2_final.py --set personalised`.

## 6. Running it

From the project folder, using the project's Python environment:

```powershell
# tests
python -m pytest -q
# interactive demo
streamlit run content/pkt/app.py
# reproduce the evaluation
python content/pkt/scripts/v2_cache_mastery.py --archive data/codeforces.zip --checkpoint content/pkt/results/ksap_hf_v1/mastery.pt --output mastery_cache.json
python content/pkt/scripts/v2_final.py --archive data/codeforces.zip --bundle content/pkt/results/ksap_hf_v1 --cache mastery_cache.json --split valid --output-dir content/pkt/results/my_valid
python content/pkt/scripts/v2_figures.py
```

`data/codeforces.zip` and `content/pkt/results/ksap_hf_v1/mastery.pt` are put in place by `setup_data.py` at the repository root.

## 7. Files added

| Path | Purpose |
|---|---|
| `content/pkt/recommendation/v2.py` | Switchable v2 components (difficulty smoothing/buckets, relative weak sets, window graph, DPM weights, hybrid/window DMF) |
| `content/pkt/recommendation/resources.py` | Concept names, curated learning material, explanation text |
| `content/pkt/scripts/v2_cache_mastery.py` | Prefix-only concept scores at five cut-offs |
| `content/pkt/scripts/v2_experiments.py` | Multi-cutoff harness, metrics, learner bootstrap |
| `content/pkt/scripts/v2_final.py` | Frozen v1/v2 + ablations; validation and test runs |
| `content/pkt/scripts/v2_figures.py` | Report figures |
| `content/pkt/app.py` | Streamlit demo |
| `content/pkt/results/ksap_v2_valid/`, `ksap_v2_test/` | `summary.csv`, `per_learner.csv`, full `results.json` |
| `content/pkt/results/ksap_v2_figures/` | agreement, trade-off and ablation figures (valid + test) |
| `tests/test_ksap_v2.py` | 13 new tests, including future-data invariance and a hand-computed NDCG |

## 8. Limitations

- **Problem IDs are internal.** The source repo provides no mapping to Codeforces problems; 23 were recovered (`results/problem_codes.json`), the rest are linked through topic and rating filters.
- **The cohort is small.** 16 / 17 / 22 learners per split, so CIs are wide and no claim of significant ranking superiority is made.
- **Offline agreement is not learning gain.** Measuring learning improvement would need a user study.
- **The concept scores are uncalibrated** (see `DIAGNOSIS_CALIBRATION.md` for the calibrated alternative).
