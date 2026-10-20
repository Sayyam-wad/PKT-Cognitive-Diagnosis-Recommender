"""Convert rows [a, b) of the PDKT C++ submissions; write conv_<a>_<b>.json. Crashes kill only this worker."""
import csv, json, sys
from paths import PKT, WORK
sys.path.insert(0, str(PKT / 'preprocessing_live'))
a, b = int(sys.argv[1]), int(sys.argv[2])
codes = json.load(open(WORK / 'cpp_codes.json'))
vocab = json.load(open(WORK / 'vocab.json'))
from cpp_to_astnn import source_to_blocks, to_ids
unk = len(vocab); out = {}
for i in range(a, min(b, len(codes))):
    try:
        out[i] = [to_ids(x, vocab, unk) for x in source_to_blocks(codes[i], "s.cpp")]
    except Exception:
        out[i] = []
json.dump(out, open(WORK / 'parts' / f'conv_{a}_{b}.json', 'w'))
