# Exploration script (16 Oct 2026); paths below point to the cloud workspace used for the feasibility study.
"""End-to-end: raw C++ -> converter -> ASTNN-Longformer (HF checkpoint) -> mastery.
import os
NODE_W2V = os.environ.get("NODE_W2V", "node_w2v_128")  # path to the dataset node_w2v_128

Also compares code embeddings of converted real code with dataset code embeddings.
"""
import json, sys, random
import numpy as np, torch
sys.path.insert(0, "/home/claude/work/ksap/content/pkt")
sys.path.insert(0, "/home/claude/work/ksap/content/pkt/preprocessing_live")
from cpp_to_astnn import source_to_blocks, to_ids
from recommendation.mastery import HFMastery
from recommendation.data import Interaction, iter_histories
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

torch.set_num_threads(2)
kv = Word2Vec.load(NODE_W2V).wv
UNK = len(kv.index_to_key)
m = HFMastery('/home/claude/work/ksap/content/pkt/results/ksap_hf_v1/mastery.pt', 'cpu')
rows = [json.loads(l) for l in open('/mnt/user-data/uploads/pkt_backup/objective3_ksap/pdkt_cpp_sample.jsonl')]
random.seed(0)

# 1) a synthetic learner: 30 real C++ submissions, tagged with plausible Codeforces concepts
conv = []
for r in rows:
    b = source_to_blocks(r['code'])
    if b:
        conv.append(([to_ids(x, kv.key_to_index, UNK) for x in b], int(r['result'])))
hist = [Interaction('live', i, 1 + i, (7, 8), y, roots, 'live', 0) for i, (roots, y) in enumerate(conv[:30])]
mastery = m.predict(hist)
print('end-to-end mastery OK:', len(mastery), 'concepts, range', round(min(mastery), 3), '-', round(max(mastery), 3))

# 2) code-embedding comparison (ASTNN encoder, BiGRU pooled) dataset vs converted real code
@torch.no_grad()
def embed(list_of_roots):
    out = []
    for roots in list_of_roots:
        code = m.model._encode_ast_batch([[roots[:30]]], 1, 1)
        out.append(code[0, 0].numpy())
    return np.array(out)
ds = []
for h in iter_histories('/home/claude/work/ksap/data/codeforces.zip', 'test'):
    ds += [e.ast_roots for e in h.interactions]
ds = random.sample(ds, 600)
cv = [r for r, _ in random.sample(conv, 600)]
E_ds, E_cv = embed(ds), embed(cv)
X = np.vstack([E_ds, E_cv]); y = np.r_[np.zeros(len(E_ds)), np.ones(len(E_cv))]
auc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring='roc_auc').mean()
cos = lambda a, b: (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
c_ds = E_ds.mean(0)
print(f'domain classifier AUC (dataset vs converted code embeddings): {auc:.3f}  (0.5 = indistinguishable)')
print(f'mean cosine to dataset centroid: dataset {np.mean([cos(e, c_ds) for e in E_ds]):.3f}  converted {np.mean([cos(e, c_ds) for e in E_cv]):.3f}')
# baseline for comparison: random token trees of similar size should look far away
rand = [[[random.randrange(UNK), [random.randrange(UNK)], [random.randrange(UNK)]] for _ in range(20)] for _ in range(200)]
E_r = embed(rand)
print(f'reference - random token trees: mean cosine to centroid {np.mean([cos(e, c_ds) for e in E_r]):.3f}')
