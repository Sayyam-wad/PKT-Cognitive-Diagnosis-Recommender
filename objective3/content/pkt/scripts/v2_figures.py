"""Report figures for the KSAP v2 evaluation (static PNGs)."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})

MAIN = ["Random", "Popularity", "DMF only (no KSAP filters)", "KSAP v1 (paper-faithful)", "KSAP v2"]
SHORT = {"Random": "Random", "Popularity": "Popularity", "DMF only (no KSAP filters)": "DMF only",
         "KSAP v1 (paper-faithful)": "KSAP v1", "KSAP v2": "KSAP v2",
         "v2 targeted (6 weakest concepts)": "KSAP v2 targeted",
         "v2 - knowledge graph (order by mastery)": "v2 − graph (weakest-first)", "v2 - DPM (no difficulty filter)": "v2 − DPM",
         "v2 - DMF (popularity ranking)": "v2 − DMF", "v2 - diagnosis (all concepts weak)": "v2 − diagnosis"}


def learner_ci(res, name, key, reps=2000, seed=0):
    vals = np.array([d[name][key] for d in res["per_learner"].values()])
    rng = np.random.default_rng(seed)
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(reps)]
    return vals.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def bars(res, split, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (key, label) in zip(axes, (("hit_rate", "HR@10"), ("ndcg_at_k", "NDCG@10"))):
        ys = np.arange(len(MAIN))[::-1]
        for y, n in zip(ys, MAIN):
            m, lo, hi = learner_ci(res, n, key)
            color = BLUE if n.startswith("KSAP") else GRAY
            ax.barh(y, m, height=0.55, color=color, zorder=2)
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
            ax.text(hi + 0.01, y, f"{m:.3f}", va="center", color=INK, fontsize=9)
        ax.set_yticks(ys, [SHORT[n] for n in MAIN])
        ax.set_title(label, color=INK, loc="left", fontsize=11)
        ax.grid(axis="x", color=GRID, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, max(ax.get_xlim()[1], 0.1) * 1.12)
    fig.suptitle(f"Agreement with learners' future attempts ({split}: {res['learners']} learners, "
                 f"{res['snapshots']} snapshots; bars = 95% learner bootstrap CI)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def tradeoff(res, split, out):
    names = MAIN[1:] + ["v2 targeted (6 weakest concepts)", "v2 - knowledge graph (order by mastery)"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for n in names:
        x, y = res["summary"][n]["ndcg_at_k"], res["summary"][n]["targets_6_weakest"]
        color = BLUE if (n.startswith("KSAP") or n.startswith("v2")) else GRAY
        ax.scatter(x, y, s=70, color=color, edgecolor=SURF, linewidth=2, zorder=3)
        dx, dy, ha = {"KSAP v1 (paper-faithful)": (7, 10, "left"),
                      "v2 targeted (6 weakest concepts)": (-9, 6, "right")}.get(n, (7, 4, "left"))
        ax.annotate(SHORT[n], (x, y), xytext=(dx, dy), textcoords="offset points", color=INK, fontsize=9, ha=ha)
    ax.set_xlabel("NDCG@10 (agreement with future attempts)")
    ax.set_ylabel("Share of recommended problems\ntagged with the learner's 6 weakest concepts")
    ax.grid(color=GRID, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Accuracy vs. pedagogical targeting ({split})", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def ablation(res, split, out):
    names = ["KSAP v2", "v2 - knowledge graph (order by mastery)", "v2 - DPM (no difficulty filter)",
             "v2 - DMF (popularity ranking)", "v2 - diagnosis (all concepts weak)"]
    keys = (("hit_rate", "HR@10"), ("ndcg_at_k", "NDCG@10"), ("targets_6_weakest", "Weak-concept share"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (key, label) in zip(axes, keys):
        ys = np.arange(len(names))[::-1]
        for y, n in zip(ys, names):
            v = res["summary"][n][key]
            ax.barh(y, v, height=0.55, color=BLUE if n == "KSAP v2" else GRAY, zorder=2)
            ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(ys, [SHORT[n] for n in names])
        ax.set_title(label, loc="left", color=INK, fontsize=11)
        ax.grid(axis="x", color=GRID, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
    fig.suptitle(f"Ablation: removing one KSAP component at a time ({split})", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    outdir = ROOT / "ksap_v2_figures"
    outdir.mkdir(exist_ok=True)
    for split in ("valid", "test"):
        res = json.loads((ROOT / f"ksap_v2_{split}" / "results.json").read_text())
        bars(res, split, outdir / f"{split}_agreement.png")
        tradeoff(res, split, outdir / f"{split}_tradeoff.png")
        ablation(res, split, outdir / f"{split}_ablation.png")
    print("saved", sorted(p.name for p in outdir.iterdir()))
