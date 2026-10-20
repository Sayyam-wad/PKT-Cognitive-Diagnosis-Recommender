"""Mastery-smoothing experiment (MASTERY_SMOOTHING.md).

  python content/pkt/scripts/mastery_smoothing_collect.py valid 40
  python content/pkt/scripts/mastery_smoothing_analyze.py valid
"""
import sys,json,time,random
import numpy as np, torch
from pathlib import Path
P=str(Path(__file__).resolve().parents[1]); sys.path.insert(0,P)
ROOT=str(Path(__file__).resolve().parents[3])
from recommendation.data import iter_histories
from recommendation.mastery import HFMastery
torch.set_num_threads(2)
class All(HFMastery):
    @torch.no_grad()
    def all_preds(self, inter):
        obs=inter[-self.context:]; L=len(obs)
        p_id=torch.ones((1,L),dtype=torch.long); conc=torch.zeros((1,L,38)); corr=torch.zeros((1,L,2)); ast=[[]]
        for s,it in enumerate(obs):
            p_id[0,s]=it.problem_id; conc[0,s,list(it.concept_ids)]=1; corr[0,s,it.correct]=1; ast[0].append(it.ast_roots[:self.max_roots])
        code=self.model._encode_ast_batch(ast,1,L)
        h=self.model.longformer(inputs_embeds=torch.cat((conc,code,corr),-1),attention_mask=torch.ones_like(p_id)).last_hidden_state[0]
        return self.model.predict_Linear(h).sigmoid()[:,1:].numpy()  # L x 37
m=All(P+'/results/ksap_hf_v1/mastery.pt','cpu')
split=sys.argv[1]; per=int(sys.argv[2]); random.seed(0)
out=[]
for h in iter_histories(ROOT+'/data/codeforces.zip',split):
    n=len(h.interactions)
    if n<30: continue
    cs=sorted(random.sample(range(20,n-1),min(per,n-21)))
    for c in cs:
        A=m.all_preds(h.interactions[:c]); B=m.all_preds(h.interactions[:c+1])
        nxt=h.interactions[c]; nn=h.interactions[c+1] if c+1<n else None
        out.append(dict(l=h.learner_id,c=c,A=A[-30:].round(4).tolist(),B=B[-30:].round(4).tolist(),
            nc=list(nxt.concept_ids),ny=nxt.correct, n2c=list(nn.concept_ids) if nn else None, n2y=nn.correct if nn else None))
    print(h.learner_id,n,flush=True)
json.dump(out,open(f'{P}/results/mastery_smoothing_{split}.json','w'))
