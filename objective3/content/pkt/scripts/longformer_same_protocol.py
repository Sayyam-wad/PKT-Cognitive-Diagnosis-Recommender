"""ASTNN-Longformer checkpoint scored with the same protocol as the KT baselines: test windows of up to 200
attempts (as in the original loader); the prediction for attempt t+1 uses attempts <= t only."""
import sys, numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1])); torch.set_num_threads(2)
from sklearn.metrics import roc_auc_score
from recommendation.data import iter_histories
from recommendation.mastery import HFMastery
M = HFMastery(sys.path[0] + '/results/ksap_hf_v1/mastery.pt', 'cpu', context=200).model
ps, ys = [], []
with torch.no_grad():
    for h in iter_histories(sys.path[0] + '/../../data/codeforces.zip', 'test'):
        it = h.interactions
        for s in range(0, len(it), 200):
            ch = it[s:s + 200]
            if len(ch) < 2: continue
            L = len(ch); conc = torch.zeros((1, L, 38)); corr = torch.zeros((1, L, 2)); ast = [[]]
            for k, e in enumerate(ch):
                conc[0, k, list(e.concept_ids)] = 1; corr[0, k, e.correct] = 1; ast[0].append(e.ast_roots[:30])
            x = torch.cat((conc, M._encode_ast_batch(ast, 1, L), corr), -1)
            for t in range(L - 1):
                hstate = M.longformer(inputs_embeds=x[:, :t + 1], attention_mask=torch.ones((1, t + 1), dtype=torch.long)).last_hidden_state[0, -1]
                tags = list(ch[t + 1].concept_ids)
                ps.append(torch.sigmoid(M.predict_Linear(hstate)[tags].mean()).item()); ys.append(ch[t + 1].correct)
        print(h.learner_id, len(ys), flush=True)
print(f"RESULT ASTNN-Longformer, same protocol as baselines: AUC {roc_auc_score(ys, ps):.4f} on {len(ys)} predictions")
