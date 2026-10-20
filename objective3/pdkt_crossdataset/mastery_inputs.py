"""Step 5: diagnosis inputs for KSAP on PDKT (written to PDKT_WORK): training/validation mastery and test cut-off cache."""
import sys, json
from paths import HERE, PKT, OBJ2, WORK
sys.path.insert(0, str(PKT))
import torch; torch.set_num_threads(2)
from recommendation.data import iter_histories
from recommendation.mastery import HFMastery
from scripts.v2_cache_mastery import cutoffs_for
A = HERE / 'pdkt.zip'
m = HFMastery(PKT / 'results' / 'ksap_hf_v1' / 'mastery.pt', 'cpu')   # smoothing="last" (reported setting)
for split, out in (('train', 'training_mastery.json'), ('valid', 'valid_full_mastery.json')):
    json.dump({h.learner_id: m.predict(h.interactions) for h in iter_histories(A, split)}, open(WORK / out, 'w'))
    print(split, 'done', flush=True)
cache = {}
for h in iter_histories(A, 'test'):
    for c in cutoffs_for(len(h.interactions)):
        cache[f"test|{h.learner_id}|{c}"] = m.predict(h.interactions[:c])
json.dump(cache, open(WORK / 'mastery_cache.json', 'w')); print('test cache', len(cache), flush=True)
