"""Evaluate the 0.8361 checkpoint with the ORIGINAL code (models/astnn/model_longformer.py, run.test,
load_data_astnn) exactly as main_metrics.py does, on any dataset file in the same 8-line format."""
import sys, types, argparse, numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'objective2'))
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import f1_score, mean_squared_error
from models.astnn.load_data_astnn import DATA_astnn
from models.astnn.run import test
from models.astnn.model_longformer import MODEL
test_file, w2v_path, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
# parameters exactly as main_metrics.py (codeforces branch + params_override), but on CPU
p = types.SimpleNamespace(gpu=-1, batch_size=1, concept_embed_dim=64, max_len=55, ast_dim=128,
    num_concepts=37, num_problems=7152, seqlen=200, hidden_dim=64, hidden_layers=2,
    ff_dropout=0.22469715844352778, attn_dropout=0.25004466156010574, gru_dropout=0.2,
    embedding_dropout=0.3383136695767353, attention_window=8)
wv = Word2Vec.load(w2v_path).wv
emb = np.zeros((wv.vectors.shape[0] + 1, 128), dtype="float32"); emb[:wv.vectors.shape[0]] = wv.vectors[:, :128]
p.max_tokens = wv.vectors.shape[0] + 1; p.ast_embed_dim = p.ast_encode_dim = 128
torch.manual_seed(0); np.random.seed(0)
model = MODEL(num_concepts=p.num_concepts, num_problems=p.num_problems, hidden_dim=p.hidden_dim,
    hidden_layers=p.hidden_layers, concept_embed_dim=p.concept_embed_dim, ast_embed_dim=p.ast_embed_dim,
    max_tokens=p.max_tokens, ast_encode_dim=p.ast_encode_dim, ast_pretrained_weight=emb, max_len=p.max_len,
    batch_size=p.batch_size, gpu=p.gpu, ff_dropout=p.ff_dropout, attn_dropout=p.attn_dropout,
    gru_dropout=p.gru_dropout, embedding_dropout=p.embedding_dropout, attention_window=p.attention_window)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
data = DATA_astnn(num_concepts=p.num_concepts, seqlen=p.seqlen)
arrays = data.load_data(test_file)
from sklearn.metrics import roc_auc_score, accuracy_score
out = test(model, p, *arrays)
y_true, y_score = np.asarray(out[-2]), np.asarray(out[-1])
auc = roc_auc_score(y_true, y_score); acc = accuracy_score(y_true, (y_score >= 0.5).astype(int))
f1 = f1_score(y_true, (y_score >= 0.5).astype(int)); mse = mean_squared_error(y_true, y_score)
print("\nrun.test metrics:", out[1] if len(out) == 4 else out[1:3])
print(f"RESULT  {test_file}\n  predictions {len(y_true)}  positive rate {np.mean(y_true):.3f}\n"
      f"  AUC {auc:.4f}  ACC {acc:.4f}  F1 {f1:.4f}  MSE {mse:.4f}")
