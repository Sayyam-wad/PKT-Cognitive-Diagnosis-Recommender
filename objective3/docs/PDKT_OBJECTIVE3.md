# Objective 3 on a second dataset (PDKT)

18 October 2026. The complete framework (diagnosis → KSAP → recommendations) was run on PDKT with **every setting frozen from Codeforces**. Nothing was tuned on PDKT, so this is an external check of the framework.

## Setup
- **Data.** PDKT C++ submissions converted with the validated converter (`pdkt_crossdataset/README.md`): 584 learners, 19,527 submissions, 35.6% accepted. Learners were split by learner into 233 train / 117 validation / 234 test (seed 0, `make_split.py`, `pdkt.zip`). No learner appears in two splits.
- **Diagnosis.** The original checkpoint (`best_longformer_model.pt`), unchanged, with the reported last-step setting (`mastery_inputs.py`).
- **Recommenders and protocol.** Identical to the reported Codeforces test run (`scripts/v2_final.py --set main / --set personalised`):
  - KSAP's graph, difficulties and DMF are fitted on train + validation learners;
  - each test learner is evaluated at 5 cut-offs (50–90% of their history);
  - relevant = problems they attempted later and had not attempted before.
- **Size.** 202 test learners have such future problems (770 snapshots), against 22 on Codeforces.

## Results (test, 202 learners)

| Method | HR@10 | P@10 | NDCG@10 | share on own 6 weakest |
|---|---|---|---|---|
| Random | 0.120 | 0.014 | 0.025 | 0.13 |
| Popularity | 0.233 | 0.035 | 0.083 | 0.22 |
| **DMF only** (similar learners, no KSAP filters) | **0.638** | **0.167** | **0.396** | 0.20 |
| KSAP v1 (paper-faithful) | 0.253 | 0.042 | 0.081 | 0.39 |
| KSAP v2 | 0.258 | 0.042 | 0.082 | 0.39 |
| KSAP v2-P (personalised prerequisites) | 0.257 | 0.042 | 0.081 | 0.39 |
| v2 without difficulty filter (ablation) | 0.329 | 0.055 | 0.104 | 0.40 |
| v2 targeted (6 weakest) | 0.329 | 0.054 | 0.104 | 0.50 |
| v2 ordered by mastery only (no graph) | 0.250 | 0.042 | 0.081 | 0.73 |

Learner-bootstrap 95% CIs for KSAP v2:
- **vs popularity:** HR +0.024 [−0.031, +0.082] (n.s.); NDCG −0.001 (n.s.); targeting **+0.160 [+0.126, +0.198]**.
- **vs DMF only:** HR **−0.381 [−0.444, −0.313]**; NDCG **−0.314**; targeting **+0.184 [+0.153, +0.218]**.
- **v2-P vs v2:** no meaningful difference on PDKT.
- KSAP lists average 8.3 problems; the filters sometimes leave fewer than 10, or none.

**Diagnosis on PDKT** (`diagnosis_test.txt`):
- item AUC@10 0.593 and AUC@50 0.583, similar to Codeforces (0.550 / 0.570);
- within-learner concept ranking ρ = −0.004, essentially zero (Codeforces +0.107).

## What it shows
1. **The weak-concept targeting generalises.** On a second dataset with ten times more test learners, KSAP again aims significantly more of its recommendations at each learner's weakest concepts than popularity or DMF (0.39 vs 0.22 / 0.20). The confidence intervals are tight.
2. **On "predict the next attempts" metrics, KSAP only matches popularity, and DMF is far ahead.**
   - PDKT is a course judge: learners largely work through the same assigned problems, so "what similar learners did next" (DMF) predicts the next attempts very well.
   - KSAP's concept order and difficulty filter move away from that sequence. Removing the difficulty filter, or using only the 6 weakest concepts, recovers part of the gap (HR 0.329).
   - This is the relevance-vs-remediation trade-off seen on Codeforces, in its strongest form.
3. **The diagnosis does not separate concepts on PDKT.** Its scores predict success overall but do not rank a learner's own topics. This is likely because of the hand-mapped topics (17% fall back to "implementation"), the introductory problems (few distinct topics per learner, 8 per list) and the domain shift. KSAP's concept choices on PDKT should therefore be read as weakly informed.
4. **v2-P makes no difference here,** because with few distinct topics per learner there are few prerequisites to skip.

## Caveats
- The topic mapping from PDKT's Chinese labels to the Codeforces topics is a judgement call (`pdkt_crossdataset/mapping.py`).
- Submission order follows file order; the data has no timestamps.
- Only C++ submissions were used; most PDKT activity is in C.
- The CI helper now skips learners for whom a metric is undefined (empty KSAP lists, which occur on PDKT). Codeforces results are unaffected; they have no such learners.

## Files
`pdkt_crossdataset/pdkt.zip` (the split data, 2 MB), `make_split.py`, `mastery_inputs.py`; `content/pkt/results/pdkt_ksap_test_main/` and `.../pdkt_ksap_test_personalised/` (summary, per-learner metrics, full results, diagnosis).
