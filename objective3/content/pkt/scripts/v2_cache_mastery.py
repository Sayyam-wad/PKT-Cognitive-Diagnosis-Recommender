"""Cache frozen ASTNN-Longformer concept scores at several observed cutoffs.

Only the observed prefix is passed to the model, so a cached vector at cutoff c
depends on interactions [0, c) only. Cutoffs are fractions of each learner's
record length; they are shared by every recommender evaluated in v2.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

PKT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKT_ROOT))
from recommendation.data import iter_histories  # noqa: E402
from recommendation.mastery import HFMastery  # noqa: E402

FRACTIONS = (0.5, 0.6, 0.7, 0.8, 0.9)


def cutoffs_for(n, fractions=FRACTIONS):
    out = []
    for f in fractions:
        c = max(1, min(n - 1, int(n * f)))
        if c not in out:
            out.append(c)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["valid", "test"])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--smoothing", choices=("last", "ema", "calibrated"), default="last",
                   help="'last' reproduces the reported results; 'ema' smoothed; 'calibrated' needs --calibration")
    p.add_argument("--calibration", type=Path, help="calibration.json from fit_calibration.py")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model = HFMastery(args.checkpoint, "cpu", smoothing="ema" if args.smoothing == "ema" else "last")
    cal = None
    if args.smoothing == "calibrated":
        from recommendation.calibration import CalibratedDiagnosis
        cal = CalibratedDiagnosis(args.calibration)
    cache = json.loads(args.output.read_text()) if args.output.exists() else {}
    for split in args.splits:
        for h in iter_histories(args.archive, split):
            n = len(h.interactions)
            if n < 2:
                continue
            for c in cutoffs_for(n):
                key = f"{split}|{h.learner_id}|{c}"
                if key in cache:
                    continue
                t = time.time()
                prefix = h.interactions[:c]
                cache[key] = cal.predict(prefix, *model.predict_both(prefix)) if cal else model.predict(prefix)
                print(f"{key} n={n} {time.time() - t:.1f}s", flush=True)
            args.output.write_text(json.dumps(cache))
    print("cached", len(cache))


if __name__ == "__main__":
    main()
