"""Held-out exact-match check (see OPTION_C_FEASIBILITY.md section 5). Run build_index first, then match.py <processes>."""
import json,sys,pickle
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[1])
sys.path.insert(0,ROOT+'/content/pkt'); sys.path.insert(0,ROOT+'/validation')
import validate_converter as V
keys=V.load_vocab()
rec=json.load(open(ROOT+'/validation/recovered_problem_ids.json'))
pid2code={int(k):v for k,v in rec.items()}
def s(n): return '('+' '.join([n[0]]+[s(c) for c in n[1:]])+')' if len(n)>1 else n[0]
idx={}  # code -> {sig: [(split,learner,i,correct)]}
for sp in ('train','valid','test'):
    for h in V.iter_histories(V.ARCHIVE,sp):
        for i,it in enumerate(h.interactions):
            if it.problem_id in pid2code:
                want=[V.decode(r,keys) for r in it.ast_roots if r!=[-1]]
                sig='\n'.join(s(b) for b in want)
                idx.setdefault(pid2code[it.problem_id],{}).setdefault(sig,[]).append((sp,h.learner_id,i,it.correct))
pickle.dump(idx,open(ROOT+'/validation/heldout_index.pkl','wb'))
print(sum(len(v2) for v in idx.values() for v2 in v.values()),'records,', sum(len(v) for v in idx.values()),'distinct trees')
