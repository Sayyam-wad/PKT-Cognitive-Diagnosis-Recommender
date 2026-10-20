"""How much do the code, the verdict and the tags of one attempt move the diagnosis? (OPTION_C_FEASIBILITY.md 5c)

    python content/pkt/scripts/code_effect_ablation.py      # ~10 min, 204 validation attempts
"""
import sys,json,random,numpy as np
from dataclasses import replace
from pathlib import Path
PKT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PKT))
import torch; torch.set_num_threads(2)
from recommendation.data import iter_histories
from recommendation.mastery import HFMastery
from recommendation.calibration import evidence_adjust
m=HFMastery(PKT/'results'/'ksap_hf_v1'/'mastery.pt','cpu')
H=list(iter_histories(PKT.parents[1]/'data'/'codeforces.zip','valid'))
pool=[e for h in H for e in h.interactions]
random.seed(0); rows=[]
for h in H:
    n=len(h.interactions)
    if n<30: continue
    for c in random.sample(range(20,n),min(12,n-20)):
        pre=list(h.interactions[:c]); e=h.interactions[c]
        same=[x for x in pool if x.correct==e.correct and x.learner_id!=h.learner_id]
        other=random.choice(same)
        empty=[[-1]]
        variants={'real':e,'code swapped (same verdict, same tags)':replace(e,ast_roots=other.ast_roots),
                  'code removed (same verdict, same tags)':replace(e,ast_roots=[[1]]),
                  'verdict flipped (same code)':replace(e,correct=1-e.correct),
                  'tags swapped (same code)':replace(e,concept_ids=other.concept_ids)}
        out={}
        for k,v in variants.items():
            last,ema=m.predict_both(pre+[v]); out[k]=(np.array(last),np.array(ema),np.array(evidence_adjust(ema,pre+[v])))
        r={k:[float(np.abs(out[k][j]-out['real'][j]).mean()) for j in range(3)] for k in variants if k!='real'}
        base=m.predict_both(pre); r['adding the attempt at all']=[float(np.abs(out['real'][0]-np.array(base[0])).mean()),float(np.abs(out['real'][1]-np.array(base[1])).mean()),float(np.abs(out['real'][2]-np.array(evidence_adjust(base[1],pre))).mean())]
        rows.append(r)
    print(h.learner_id,len(rows),flush=True)
keys=list(rows[0])
print(f"\nmean |change| over 37 concepts vs the real attempt   (last-step / smoothed / smoothed+record), {len(rows)} attempts")
for k in keys: print(f"  {k:42s} "+"  ".join(f"{np.mean([r[k][j] for r in rows]):.3f}" for j in range(3)))
