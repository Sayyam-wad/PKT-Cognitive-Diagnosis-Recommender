"""Validation-only exploration: KSAP settings on top of the calibrated diagnosis (DIAGNOSIS_CALIBRATION.md).

    python content/pkt/scripts/v2c_validation_grid.py
"""
import sys,json
from dataclasses import replace
from pathlib import Path
PKT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PKT))
R=PKT/'results'/'ksap_calibrated'
from recommendation import v2
from recommendation.data import iter_histories
from scripts.v2_experiments import evaluate
A=PKT.parents[1]/'data'/'codeforces.zip'
train=list(iter_histories(A,'train',include_asts=False)); valid=list(iter_histories(A,'valid',include_asts=False))
tm=json.loads((R/'training_mastery.json').read_text()); mm=[sum(v[i] for v in tm.values())/len(tm) for i in range(37)]
art=v2.fit(train,mm,dmf_checkpoint=str(PKT/'results'/'ksap_hf_v1'/'dmf.pt'),windows=(100,))
V2=v2.Config(name="v2",graph="window",window=100)
C=[V2]
for k in (8,10,12,15,20):
    for t in (1,2,3):
        C.append(replace(V2,name=f"rel{k} min5 top{t}",weak_mode="relative",weak_k=k,weak_min_candidates=5,top_k_per_concept=t))
for thr in (0.40,0.45):
    C.append(replace(V2,name=f"thr{thr} min5",weak_threshold=thr,weak_min_candidates=5))
res=evaluate(art,C,valid,json.loads((R/'mastery_cache.json').read_text()),'valid')
print(f"{'method':20s} HR     NDCG   Weak6  ExpS  n")
for n,r in res['summary'].items(): print(f"{n:20s} {r['hit_rate']:.3f}  {r['ndcg_at_k']:.3f}  {r['targets_6_weakest']:.2f}  {r['expected_success']:.3f} {r['returned']:.1f}")
