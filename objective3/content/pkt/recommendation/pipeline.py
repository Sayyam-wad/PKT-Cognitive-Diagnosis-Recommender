"""Fit and serve the offline Codeforces adaptation of KSAP."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from recommendation.dmf import train_dmf, load_dmf
from recommendation.knowledge_graph import fit_graph, weak_sequence
from recommendation.mastery import HFMastery, HF_SHA256, digest
from recommendation.preferences import fit_catalog, difficulty_preference
from recommendation.ranker import recommend


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_pipeline(histories, checkpoint, output, device="cpu", dmf_epochs=30, seed=0,
                 context=200, attention_window=8):
    histories = list(histories)
    if not histories or any(h.source_split not in ("train", "train1") for h in histories):
        raise ValueError("fit requires training histories only")
    if len({h.learner_id for h in histories}) != len(histories):
        raise ValueError("training learner IDs must be unique")
    catalog = fit_catalog(histories)
    if not catalog:
        raise ValueError("training histories contain no problems")
    provider = HFMastery(checkpoint, device, context=context, attention_window=attention_window)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mastery = [provider.predict(h.interactions) for h in histories]
    graph = fit_graph(histories, np.mean(mastery, axis=0).tolist())
    write_json(output / "catalog.json", catalog)
    write_json(output / "graph.json", graph)
    write_json(output / "training_mastery.json", {
        h.learner_id: values for h, values in zip(histories, mastery)
    })
    print(f"Fitting DMF on {len(histories)} training learners and {len(catalog)} problems", flush=True)
    dmf = train_dmf(histories, sorted(int(pid) for pid in catalog), epochs=dmf_epochs,
                    seed=seed, device=device)
    torch.save(dmf, output / "dmf.pt")
    shutil.copyfile(checkpoint, output / "mastery.pt")
    files = ("catalog.json", "graph.json", "training_mastery.json", "dmf.pt", "mastery.pt")
    manifest = {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "KSAP-Codeforces", "training_learners": [h.learner_id for h in histories],
        "training_interactions": sum(len(h.interactions) for h in histories),
        "catalog_size": len(catalog), "seed": seed, "dmf_epochs": dmf_epochs,
        "context": context, "max_roots": 30, "attention_window": attention_window,
        "checkpoint_sha256": provider.sha256, "is_shared_hf_checkpoint": provider.sha256 == HF_SHA256,
        "checkpoint_training_provenance": "objective2/models/astnn/main.py (ASTNN-Longformer, Codeforces)",
        "mastery_calibration": "uncalibrated concept-head probabilities",
        "dmf_objective": dmf["objective"], "dmf_loss_first": dmf["history"][0],
        "dmf_loss_last": dmf["history"][-1],
        "torch": str(torch.__version__), "numpy": np.__version__,
        "files_sha256": {name: digest(output / name) for name in files},
        "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    }
    # A manifest is written last; an interrupted fit is not a loadable bundle.
    write_json(output / "manifest.json", manifest)
    return manifest


class KSAPRecommender:
    def __init__(self, bundle, device="cpu"):
        bundle = Path(bundle)
        self.manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 1:
            raise ValueError("unsupported KSAP bundle version")
        expected = {"catalog.json", "graph.json", "training_mastery.json", "dmf.pt", "mastery.pt"}
        if set(self.manifest.get("files_sha256", {})) != expected:
            raise ValueError("incomplete KSAP bundle manifest")
        for name, checksum in self.manifest["files_sha256"].items():
            if digest(bundle / name) != checksum:
                raise ValueError(f"KSAP bundle checksum mismatch: {name}")
        self.catalog = json.loads((bundle / "catalog.json").read_text(encoding="utf-8"))
        self.graph = json.loads((bundle / "graph.json").read_text(encoding="utf-8"))
        self.mastery = HFMastery(bundle / "mastery.pt", device,
                                 context=self.manifest["context"],
                                 max_roots=self.manifest["max_roots"],
                                 attention_window=self.manifest["attention_window"])
        self.dmf = load_dmf(bundle / "dmf.pt", device=device)
        if set(map(int, self.catalog)) != set(self.dmf.problem_ids):
            raise ValueError("DMF and catalog vocabularies differ")

    def recommend(self, interactions, learner_id="new_learner", top_k_per_concept=3,
                  limit=10, allow_fallback=False):
        observed = tuple(interactions)
        values = self.mastery.predict(observed)
        sequence = weak_sequence(self.graph, values)
        preference = difficulty_preference(observed, self.catalog)
        attempted = {event.problem_id for event in observed}
        scores = self.dmf.score(self.dmf.problem_ids, attempted)
        ranking = recommend(self.catalog, sequence["ordered_concepts"], preference, scores,
                            attempted, values, self.graph, top_k_per_concept, limit, allow_fallback)
        return {
            "learner_id": learner_id, "observed_interactions": len(observed),
            "model_context_interactions": min(len(observed), self.mastery.context),
            "mastery": {str(i): p for i, p in enumerate(values, 1)},
            **sequence, "difficulty_preference": preference, **ranking,
            "mastery_checkpoint_sha256": self.mastery.sha256,
            "checkpoint_training_provenance": self.manifest["checkpoint_training_provenance"],
            "note": "Internal problem IDs; no verified external URLs or learning-gain claim.",
        }
