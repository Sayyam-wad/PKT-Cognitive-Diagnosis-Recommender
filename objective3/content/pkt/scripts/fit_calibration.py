"""Fit the calibrated concept diagnosis on TRAINING learners only (DIAGNOSIS_CALIBRATION.md).

    python content/pkt/scripts/fit_calibration.py --archive data/codeforces.zip \
        --checkpoint content/pkt/results/ksap_hf_v1/mastery.pt --output-dir content/pkt/results/ksap_calibrated

Writes calibration.json (weights), training_mastery.json and valid_full_mastery.json
(full-history calibrated vectors used to fit KSAP's global artefacts). ~10 minutes on CPU.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PKT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKT))
from recommendation.calibration import CalibratedDiagnosis, concept_features  # noqa: E402
from recommendation.data import iter_histories  # noqa: E402
from recommendation.mastery import HFMastery  # noqa: E402

HORIZON = 10      # target: the next 10 attempts carrying the concept
PER_LEARNER = 80  # cut-offs sampled per training learner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model = HFMastery(args.checkpoint, "cpu")
    rows, ys, owners = [], [], []
    train = list(iter_histories(args.archive, "train"))
    for h in train:
        n = len(h.interactions)
        for c in range(10, n - 1, max(5, (n - 10) // PER_LEARNER)):
            last, ema = model.predict_both(h.interactions[:c])
            X = concept_features(h.interactions[:c], last, ema)
            for e in h.interactions[c:c + HORIZON]:
                for t in e.concept_ids:
                    rows.append(X[t - 1]); ys.append(e.correct); owners.append(h.learner_id)
        print(f"{h.learner_id}: {n} attempts", flush=True)
    counts = {o: owners.count(o) for o in set(owners)}
    w = np.array([1 / counts[o] for o in owners]); w *= len(w) / w.sum()   # each learner weighs the same
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    pipe.fit(np.array(rows), np.array(ys), logisticregression__sample_weight=w)
    params = CalibratedDiagnosis.params_from_sklearn(pipe)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration.json").write_text(json.dumps(params, indent=1))
    cal = CalibratedDiagnosis(params)
    for split, name in (("train", "training_mastery.json"), ("valid", "valid_full_mastery.json")):
        hs = train if split == "train" else list(iter_histories(args.archive, "valid"))
        vec = {h.learner_id: cal.predict(h.interactions, *model.predict_both(h.interactions)) for h in hs}
        (args.output_dir / name).write_text(json.dumps(vec))
    print(f"fitted on {len(ys)} rows from {len(counts)} training learners -> {args.output_dir}")


if __name__ == "__main__":
    main()
