# Objective 2: ASTNN + Longformer knowledge tracing

The model reads a learner's submission history. Each submission contributes three inputs:

- its concept tags;
- its code, as ASTNN statement trees encoded by a tree encoder and a bidirectional GRU;
- its verdict.

A Longformer (local attention window 8) predicts success on the next attempt, with one logit per concept (37 concepts). The ASTNN + LSTM model is the baseline: the same encoder with an LSTM in place of the Longformer.

## Files

| File | Use |
|---|---|
| `models/astnn/model_longformer.py` | ASTNN + Longformer model |
| `models/astnn/model_lstm.py` | ASTNN + LSTM baseline |
| `models/astnn/load_data_astnn.py` | Reader for the 8-line ASTNN records (windows of 200 attempts) |
| `models/astnn/run.py` | Training and evaluation loops |
| `models/astnn/main.py` | Trains the Longformer model with the tuned hyper-parameters; saves `results/best_longformer_model.pt` |
| `models/astnn/main_metrics.py` | Evaluates `results/best_longformer_model.pt` on the test split (AUC, ACC, F1, MSE, ROC curve) |
| `models/astnn/main_lstm.py` | Trains the LSTM baseline (pass `--dataset codeforces`); saves to `results/pdkt_lstm/best_lstm_model.pt` |
| `models/astnn/main_lstm_metrics.py` | Evaluates `best_lstm_model.pt` from the current folder |
| `results/best_longformer_model.pt` | Trained ASTNN + Longformer (SHA-256 b4be633b…96a4) |
| `results/best_lstm_model.pt` | Trained ASTNN + LSTM (SHA-256 a2e212d0…5b3f) |

## Settings (Codeforces)

| Setting | Value |
|---|---|
| Concepts | 37 |
| Problems | 7,152 |
| Sequence length | 200 |
| Hidden size and layers | 64, 2 |
| AST embedding | 128 (`node_w2v_128`) |
| Tuned Longformer hyper-parameters (Optuna) | learning rate 1.80e-4, feed-forward dropout 0.225, attention dropout 0.250, embedding dropout 0.338, attention window 8, maximum statement trees per attempt 55 |
| Training | 50 epochs; the best validation checkpoint is kept |

## Run

From this folder, after `python setup_data.py` at the repository root:

```bash
python -m models.astnn.main            # train (GPU 0 by default; --gpu -1 for CPU)
python -m models.astnn.main_metrics    # evaluate
cp results/best_lstm_model.pt . && python -m models.astnn.main_lstm_metrics    # evaluate the baseline
```

## Results (Codeforces test split, 6,445 predictions)

| Model | AUC | ACC | F1 | MSE |
|---|---|---|---|---|
| ASTNN + LSTM | 0.7035 | 0.6577 | 0.6806 | 0.2215 |
| **ASTNN + Longformer** | **0.8361** | **0.7528** | **0.7594** | **0.1680** |

Zero-shot on the PDKT dataset (no retraining, 18,934 predictions), the Longformer reaches AUC 0.7700 against 0.6161 for the LSTM. See `../objective3/pdkt_crossdataset/README.md`.
