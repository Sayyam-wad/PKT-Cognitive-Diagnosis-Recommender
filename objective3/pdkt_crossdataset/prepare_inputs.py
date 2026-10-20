"""Step 1: extract the C++ submissions' source code and the Codeforces token vocabulary into PDKT_WORK."""
import csv, json, sys
from paths import RAW, WORK, W2V
from gensim.models.word2vec import Word2Vec
csv.field_size_limit(10**9)
rows = [r for r in csv.DictReader(open(RAW / 'user_sequence.csv', encoding='utf-8', errors='replace')) if r['language'] == 'C++']
json.dump([r['code'] for r in rows], open(WORK / 'cpp_codes.json', 'w'))
json.dump(Word2Vec.load(str(W2V)).wv.key_to_index, open(WORK / 'vocab.json', 'w'))
print(f"{len(rows)} C++ submissions -> {WORK}")
