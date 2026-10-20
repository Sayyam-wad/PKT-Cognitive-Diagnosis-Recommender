"""Held-out exact-match check (see OPTION_C_FEASIBILITY.md section 5). Run build_index first, then match.py <processes>."""
import sys,pickle,json,os
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[1])
sys.path.insert(0,ROOT+'/content/pkt/preprocessing_live')
from multiprocessing import Pool
import pyarrow.parquet as pq
idx=pickle.load(open(ROOT+'/validation/heldout_index.pkl','rb'))
sys.path.insert(0,ROOT+'/content/pkt'); sys.path.insert(0,ROOT+'/validation')
import validate_converter as V
keys=V.load_vocab()  # node_vocab.json if present, else gensim
vocab=set(keys)
def s(n): return '('+' '.join([n[0]]+[s(c) for c in n[1:]])+')' if len(n)>1 else n[0]
def mu(n): return [n[0] if n[0] in vocab else '<UNK>']+[mu(c) for c in n[1:]]
def work(row):
    sid,code,src=row
    try:
        from cpp_to_astnn import source_to_blocks
        sig='\n'.join(s(mu(b)) for b in source_to_blocks(src,'sub.cpp'))
    except Exception as e:
        return (sid,code,'ERR')
    return (sid,code,idx.get(code,{}).get(sig))
if __name__=='__main__':
    t=pq.read_table(ROOT+'/validation/candidates.parquet').to_pydict()
    rows=[(sid,f"{c}{p}",src) for sid,c,p,src in zip(t['submission_id'],t['contestId'],t['problem_index'],t['source']) if src]
    print(len(rows),flush=True)
    out=open(ROOT+'/validation/heldout_matches.jsonl','w'); n=err=0
    with Pool(int(sys.argv[1])) as pool:
        for k,(sid,code,m) in enumerate(pool.imap_unordered(work,rows,chunksize=50)):
            if m=='ERR': err+=1
            elif m: out.write(json.dumps({'sid':sid,'code':code,'records':m})+'\n'); out.flush(); n+=1
            if k%5000==0: print(k,'done; matches',n,'errors',err,flush=True)
    print('FINISHED matches',n,'errors',err,flush=True)
