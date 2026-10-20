# Cross-dataset check: the Codeforces checkpoints on PDKT

18 October 2026. The original checkpoint (`results/best_longformer_model.pt`, SHA-256 b4be633b…) was applied, **without retraining**, to the PDKT C++ submissions. It was evaluated with the original code (`models/astnn/model_longformer.py`, `run.test`, `load_data_astnn.py`, byte-identical copies) and the settings of `main_metrics.py`. No original file was modified.

## Reproduction check first
The same wrapper (`evaluate_original.py`) on the Codeforces test file gives **AUC 0.8361, ACC 0.7528, F1 0.7594, MSE 0.1680**, identical to the reported results. The pipeline is therefore faithful.

## Result

Both checkpoints were evaluated with their own original code and settings. `evaluate_original.py` follows `main_metrics.py` for the Longformer; `evaluate_lstm.py` follows `main_lstm_metrics.py` for the LSTM baseline (`model_lstm.py`, `best_lstm_model.pt`, SHA-256 a2e212d0…). The PDKT test file is the same for both.

| Model | Dataset | Predictions | AUC | ACC | F1 | MSE |
|---|---|---|---|---|---|---|
| ASTNN + LSTM (baseline) | Codeforces test | 6,445 | 0.7035 | 0.6577 | 0.6806 | 0.2215 |
| **ASTNN + Longformer** | Codeforces test | 6,445 | **0.8361** | **0.7528** | **0.7594** | **0.1680** |
| ASTNN + LSTM (baseline) | PDKT, zero-shot | 18,934 | 0.6161 | 0.5338 | 0.4909 | 0.2459 |
| **ASTNN + Longformer** | PDKT, zero-shot | 18,934 | **0.7700** | **0.7497** | **0.5933** | **0.1835** |

- PDKT: 584 learners, 34.9% of predictions are "accepted". Codeforces: 22 learners, 49.9%.
- On the unseen PDKT data the Longformer model is ahead of the LSTM baseline by 0.154 AUC, about the same gap as on Codeforces (0.133).
- PDKT is imbalanced, so always predicting "wrong" gives 65% accuracy. The LSTM's 53% accuracy at the 0.5 threshold is below that; AUC is the comparable metric.

## How the PDKT data was prepared (from scratch)
1. **Submissions.** From `used_in_pdkt/user_sequence.csv`, C++ submissions only (19,733 of 75,993; the model was trained on GNU C++). They were kept in file order per user. Users with at least 3 submissions: 584.
2. **Code to trees.** The validated converter (`content/pkt/preprocessing_live/cpp_to_astnn.py`) was used: libclang 13, no standard headers, macros expanded. It matches the Codeforces dataset exactly on 48 real submissions. Tokens were mapped with the Codeforces `node_w2v_128` vocabulary; 1.7% fall outside it.
   - 9 files crash or hang the parser and were skipped (`supervise.py`).
   - 99 produced no code tree and were dropped.
3. **Topics.** PDKT's `tags` column is the problem ID, not a topic. Its real labels (`problem_tags.csv`, Chinese) were mapped by hand to the 37 Codeforces topics (`mapping.py`, e.g. 贪心→greedy, 并查集→dsu, 二分查找→binary search, 模拟/循环→implementation). Problems with only language or course labels fall back to "implementation" (3,388 submissions, 17%).
4. **File.** The result was written in the same 8-line format as `astnn_test` and evaluated with sequence length 200 (`assemble.py`, `evaluate_original.py`).

## Caveats
- The topic mapping is a judgement call; a different mapping could move the result somewhat.
- PDKT is a different judge, with mostly introductory problems and a different population of learners.
- The submission order within a user follows the file order (the data has no timestamps).

## Files
`mapping.py`, `conv_worker.py` + `supervise.py` (crash-tolerant conversion), `assemble.py`, `evaluate_original.py` (Longformer), `evaluate_lstm.py` (LSTM baseline), `result_pdkt.txt`, `result_pdkt_lstm.txt`. The raw PDKT export and the generated 58 MB test file are not included. To rebuild from the raw export, set `PDKT_RAW` to its folder and run, in order: `prepare_inputs.py`, `supervise.py`, `assemble.py`, `make_split.py`, `mastery_inputs.py` (intermediate files go to `PDKT_WORK`, default `./work`). The split result, `pdkt.zip`, is included, so the Objective 3 run on PDKT needs only the last step.
