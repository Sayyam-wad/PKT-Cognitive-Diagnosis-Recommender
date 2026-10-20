# PKT Cognitive Diagnosis Recommender

Capstone project: a cognitive-diagnosis-based learning material recommender for programming.

- **Objective 2:** an ASTNN + Longformer knowledge-tracing model. It reads a learner's code submissions and estimates their mastery of 37 programming concepts. An ASTNN + LSTM model is the baseline.
- **Objective 3:** a personalised recommender (KSAP). It turns those knowledge gaps into a study plan, 10 Codeforces problems and learning material. It runs as an interactive web prototype that accepts pasted C++ code.

```
learner's submissions ──► Objective 2: ASTNN-Longformer ──► 37 concept mastery scores
                                                                    │
                       Objective 3: knowledge graph + difficulty preference + DMF
                                                                    │
                                  weak concepts, study plan, 10 problems, reading links
```

## Repository layout

| Folder | Contents |
|---|---|
| `setup_data.py` | One-time setup: downloads the dataset and puts every input in place |
| `objective2/` | Training and evaluation code for the ASTNN-Longformer and ASTNN-LSTM models, and the trained checkpoints |
| `objective3/content/pkt/` | The recommender (`recommendation/`), the live C++ converter (`preprocessing_live/`), the web app (`app.py`), experiment scripts (`scripts/`) and saved results (`results/`) |
| `objective3/tests/` | Automated tests (90) |
| `objective3/docs/` | Write-ups of every Objective 3 experiment and result |
| `objective3/pdkt_crossdataset/` | The second dataset (PDKT): conversion pipeline and the prepared split |
| `objective3/validation/` | Exact-match validation of the C++ → AST converter |

## 1. Install

Python 3.11 is recommended.

```bash
python -m venv venv
venv/Scripts/activate            # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch that matches your driver first (see pytorch.org).

## 2. Get the data

```bash
python setup_data.py
```

This downloads `codeforces.zip` (85 MB) from the original PKT repository ([pkt-attn/pkt-attn](https://github.com/pkt-attn/pkt-attn)) and checks its SHA-256. It then:

- extracts the training inputs for Objective 2;
- places the archive for Objective 3;
- copies the trained ASTNN-Longformer checkpoint into the Objective 3 bundle.

If you already have the archive, use `python setup_data.py --zip path/to/codeforces.zip`.

## 3. Objective 2: the knowledge-tracing model

Run from `objective2/`:

```bash
cd objective2
python -m models.astnn.main            # train ASTNN-Longformer (50 epochs, tuned hyper-parameters)
python -m models.astnn.main_metrics    # evaluate results/best_longformer_model.pt on the test split
python -m models.astnn.main_lstm --dataset codeforces    # train the ASTNN-LSTM baseline
```

The trained checkpoints are already in `objective2/results/`, so training is optional:

| Model | Test AUC | ACC | F1 | MSE |
|---|---|---|---|---|
| ASTNN + LSTM (baseline) | 0.7035 | 0.6577 | 0.6806 | 0.2215 |
| **ASTNN + Longformer** | **0.8361** | **0.7528** | **0.7594** | **0.1680** |

The scripts default to GPU 0. `main_metrics.py` loads the checkpoint onto the GPU. To evaluate on a CPU-only machine, run:

```bash
cd objective3/pdkt_crossdataset
python evaluate_original.py ../../objective2/data/codeforces/astnn/astnn_test ../../objective2/data/codeforces/astnn/node_w2v_128 ../../objective2/results/best_longformer_model.pt
```

That command uses the same model code and settings. If you retrain, run `python setup_data.py` again to copy the new checkpoint into Objective 3. Details are in [objective2/README.md](objective2/README.md).

## 4. Objective 3: the recommender

Run everything from `objective3/`.

**Tests**

```bash
cd objective3
python -m pytest -q
```

**Web prototype**

```bash
python -m streamlit run content/pkt/app.py
```

Pick a learner and how many of their attempts the system has seen. The page shows:

1. the per-concept diagnosis;
2. the knowledge-structure order;
3. the difficulty preference;
4. **what to work on:** weakest concepts, a study plan and reading;
5. 10 recommended problems with an explanation and learning material each.

Section 0 accepts a pasted C++ submission. It is converted into the model's AST format and updates everything live.

Optional: `python content/pkt/scripts/fetch_cf_practice.py` adds three named Codeforces practice problems to each recommendation. It needs internet access.

**Reproduce the evaluation (Codeforces test split)**

```bash
# main result: KSAP v1 / v2, baselines and ablations
python content/pkt/scripts/v2_final.py --archive data/codeforces.zip --bundle content/pkt/results/ksap_hf_v1 \
  --cache content/pkt/results/ksap_last_inputs/mastery_cache.json \
  --full-mastery content/pkt/results/ksap_last_inputs/valid_full_mastery.json \
  --split test --set main --output-dir my_results/main
# personalised recommender (KSAP v2-P): same command with --set personalised

# DKT / DKVMN / SAKT baselines, and each as the diagnosis engine inside KSAP
python content/pkt/scripts/kt_baselines.py --archive data/codeforces.zip --out my_results/kt_baselines
python content/pkt/scripts/longformer_same_protocol.py

# diagnosis quality and the calibrated diagnosis
python content/pkt/scripts/diagnosis_eval.py --split test --cache last=content/pkt/results/ksap_last_inputs/mastery_cache.json calibrated=content/pkt/results/ksap_calibrated/mastery_cache.json
```

The saved outputs of every run are in `objective3/content/pkt/results/`. The other experiments are described in the docs below.

**Second dataset (PDKT)**

`objective3/pdkt_crossdataset/pdkt.zip` holds the prepared PDKT split. To run Objective 3 on it, first compute its diagnosis inputs:

```bash
python pdkt_crossdataset/mastery_inputs.py
```

Then run `v2_final.py` with `--archive pdkt_crossdataset/pdkt.zip` and the files it wrote. See [pdkt_crossdataset/README.md](objective3/pdkt_crossdataset/README.md).

## Results and write-ups (`objective3/docs/`)

| Document | Contents |
|---|---|
| [KSAP_V2_RESULTS.md](objective3/docs/KSAP_V2_RESULTS.md) | Main recommender results, ablation, personalisation (v2-P) |
| [KT_BASELINES.md](objective3/docs/KT_BASELINES.md) | DKT, DKVMN and SAKT as knowledge-tracing models and as the diagnosis engine |
| [DIAGNOSIS_CALIBRATION.md](objective3/docs/DIAGNOSIS_CALIBRATION.md) | Calibrated concept diagnosis |
| [MASTERY_SMOOTHING.md](objective3/docs/MASTERY_SMOOTHING.md) | Smoothed mastery estimate for live input |
| [OPTION_C_FEASIBILITY.md](objective3/docs/OPTION_C_FEASIBILITY.md) | C++ → AST converter and live code input |
| [PDKT_OBJECTIVE3.md](objective3/docs/PDKT_OBJECTIVE3.md) | Objective 3 on the second dataset |

## Data sources

- Codeforces ASTNN dataset: the original PKT repository, [pkt-attn/pkt-attn](https://github.com/pkt-attn/pkt-attn). It is downloaded by `setup_data.py`, not stored here.
- PDKT: the processed split only (`pdkt.zip`). The raw export is not redistributed.
