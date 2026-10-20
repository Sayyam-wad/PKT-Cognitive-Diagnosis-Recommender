"""Mastery-smoothing experiment (MASTERY_SMOOTHING.md).

  python content/pkt/scripts/mastery_smoothing_collect.py valid 40
  python content/pkt/scripts/mastery_smoothing_analyze.py valid
"""
import json,sys,numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
P=Path(__file__).resolve().parents[1]
D=json.load(open(P/'results'/f'mastery_smoothing_{sys.argv[1]}.json'))
def ema(X,a):
    e=X[0]
    for x in X[1:]: e=a*x+(1-a)*e
    return e
ests={'raw (last step)':lambda X:X[-1]}
for k in (5,10,20,30): ests[f'mean last {k}']=(lambda k: lambda X:X[-k:].mean(0))(k)
for a in (0.3,0.1): ests[f'EMA a={a}']=(lambda a: lambda X:ema(X,a))(a)
print(f"{'estimator':18s} {'AUC next':>9s} {'AUC +1':>7s} {'off-tag |Δ|':>12s} {'on-tag |Δ|':>11s} {'max off |Δ|':>12s} {'dir ok':>7s}")
for name,f in ests.items():
    s1=[];y1=[];s2=[];y2=[];off=[];on=[];mx=[];dirok=[]
    for r in D:
        A=np.array(r['A']);B=np.array(r['B']);ea=f(A);eb=f(B)
        idx=[c-1 for c in r['nc']]
        s1.append(ea[idx].mean());y1.append(r['ny'])
        if r['n2c']: s2.append(eb[[c-1 for c in r['n2c']]].mean());y2.append(r['n2y'])
        d=np.abs(eb-ea); mask=np.ones(37,bool);mask[idx]=False
        off.append(d[mask].mean());on.append(d[idx].mean());mx.append(d[mask].max())
        sd=(eb-ea)[idx].mean(); dirok.append((sd>0)==(r['ny']==1))
    print(f"{name:18s} {roc_auc_score(y1,s1):9.3f} {roc_auc_score(y2,s2):7.3f} {np.mean(off):12.3f} {np.mean(on):11.3f} {np.mean(mx):12.3f} {np.mean(dirok):7.2f}")
print(len(D),'cutoffs')
