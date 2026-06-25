"""
LLM-EMF Recommendation Inference Script
Loads trained model, runs cross-domain recommendation
"""
import os, sys, json, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

BASE = r"D:\CR\社会计算\1"
MODEL_PATH = os.path.join(BASE, "best_val_model.pt")
FEAT_DIR = os.path.join(BASE, "features")

sys.path.insert(0, r"D:\CR\社会计算\2\LLM-ENF\训练\主实验训练")
from cdsr_model import CDSRModel

DEVICE = "cpu"
TEXT_VARIANT = "llmdesc"

# ============================================================
# 1. Load data
# ============================================================
print("Loading data...")
train = pd.read_csv(os.path.join(BASE, "train.csv"))
val = pd.read_csv(os.path.join(BASE, "val.csv"))
test = pd.read_csv(os.path.join(BASE, "test.csv"))
meta = pd.read_csv(os.path.join(BASE, "item_meta_merged.csv"))

with open(os.path.join(BASE, "item2id.json")) as f:
    item2id = json.load(f)
id2item = {int(v): k for k, v in item2id.items()}

N_ITEMS = len(item2id)
print(f"Items: {N_ITEMS}, Train interactions: {len(train):,}")

# Build domain map
domain_map = {}
for _, row in meta.iterrows():
    idx = int(row["parent_asin"])
    domain_map[idx] = int(row["domain"])

# Build item name map
item_names = {}
for _, row in meta.iterrows():
    idx = int(row["parent_asin"])
    item_names[idx] = str(row.get("title", ""))[:80]

# ============================================================
# 2. Load features
# ============================================================
print("Loading features...")
text_map = {
    "desc": "text_bge_desc_1024.npy",
    "llm": "text_bge_llm_1024.npy",
    "llmdesc": "text_bge_llmdesc_1024.npy",
}
id_feats = torch.from_numpy(np.load(os.path.join(FEAT_DIR, "item_id_lightgcn_512.npy"))).float()
img_feats = torch.from_numpy(np.load(os.path.join(FEAT_DIR, "image_features_768.npy"))).float()
tex_feats = torch.from_numpy(np.load(os.path.join(FEAT_DIR, text_map[TEXT_VARIANT]))).float()
print(f"ID:{id_feats.shape} Image:{img_feats.shape} Text:{tex_feats.shape}")

# Domain masks
movie_mask = torch.zeros(N_ITEMS, dtype=torch.bool)
book_mask = torch.zeros(N_ITEMS, dtype=torch.bool)
movie_idx = meta[meta["domain"] == 0]["parent_asin"].astype(int).tolist()
book_idx = meta[meta["domain"] == 1]["parent_asin"].astype(int).tolist()
movie_mask[movie_idx] = True
book_mask[book_idx] = True
print(f"Movie items: {movie_mask.sum().item()}  Book items: {book_mask.sum().item()}")

# ============================================================
# 3. Load model
# ============================================================
print("Loading model...")
model = CDSRModel(N_ITEMS, id_feats, img_feats, tex_feats, movie_mask, book_mask,
                  d_model=256, n_heads=4, n_layers=2, dropout=0.2, max_len=200,
                  ablation="full")
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ============================================================
# 4. Build user sequences
# ============================================================
print("Building sequences...")

def build_user_sequences(df):
    user_seqs = {}
    for uid, g in df.groupby("user_id"):
        g = g.sort_values("timestamp")
        uid = int(uid)
        sx, sy, sxy = [], [], []
        for _, row in g.iterrows():
            ts = row["timestamp"]
            iid = int(row["item_id"])
            dom = domain_map.get(iid, 1)
            if dom == 0:
                sx.append((ts, iid))
            else:
                sy.append((ts, iid))
            sxy.append((ts, iid, dom))
        user_seqs[uid] = {"sx": sx, "sy": sy, "sxy": sxy}
    return user_seqs

train_seqs = build_user_sequences(train)
val_seqs = build_user_sequences(val)
test_seqs = build_user_sequences(test)

MAX_SEQ = 200
LAMBDA1, LAMBDA2 = 0.3, 0.1

# ============================================================
# 5. Recommendation function
# ============================================================
def get_item_name(idx):
    if idx in item_names:
        title = item_names[idx]
        if domain_map.get(idx, 1) == 0:
            return f"[Movie] {title}"
        else:
            return f"[Book]  {title}"
    return f"Item_{idx}"

def pad_seq(ids):
    ids = ids[-MAX_SEQ:]
    n = len(ids)
    return (torch.tensor(ids + [0]*(MAX_SEQ-n), dtype=torch.long).unsqueeze(0),
            torch.tensor([1]*n + [0]*(MAX_SEQ-n), dtype=torch.long).unsqueeze(0))

@torch.no_grad()
def get_scores(user_id):
    """Get movie and book scores for a user"""
    ctx = train_seqs.get(user_id, {"sx":[], "sy":[], "sxy":[]})
    vs = val_seqs.get(user_id, {"sx":[], "sy":[], "sxy":[]})
    sx_list = ctx["sx"] + vs["sx"]
    sy_list = ctx["sy"] + vs["sy"]
    sxy_list = ctx["sxy"] + vs["sxy"]

    if len(sxy_list) < 1:
        return None, None, sxy_list

    sx_ids = [x[1] for x in sx_list]
    sy_ids = [y[1] for y in sy_list]
    sxy_ids = [z[1] for z in sxy_list]

    sx_t, sx_m = pad_seq(sx_ids)
    sy_t, sy_m = pad_seq(sy_ids)
    sxy_t, sxy_m = pad_seq(sxy_ids)

    out = model(sx_t, sx_m, sy_t, sy_m, sxy_t, sxy_m, return_cross=True)

    # Movie: P_X + lambda1*P_Y_to_X + lambda2*P_XY_to_X
    movie_scores = out["P_X"][0] + LAMBDA1*out["P_Y_to_X"][0] + LAMBDA2*out["P_XY_to_X"][0]
    # Book: P_X_to_Y + lambda1*P_Y + lambda2*P_XY_to_Y
    book_scores = out["P_X_to_Y"][0] + LAMBDA1*out["P_Y"][0] + LAMBDA2*out["P_XY_to_Y"][0]

    # Exclude interacted items
    interacted = set(sxy_ids)
    movie_scores[list(interacted)] = float('-inf')
    book_scores[list(interacted)] = float('-inf')

    return movie_scores, book_scores, sxy_ids

def recommend(user_id, k=10, domain="all"):
    """Top-k recommendations. domain: 'all', 'movie', 'book'"""
    movie_scores, book_scores, _ = get_scores(user_id)
    if movie_scores is None:
        return None

    if domain == "movie":
        _, topk = torch.topk(movie_scores, k)
    elif domain == "book":
        _, topk = torch.topk(book_scores, k)
    else:
        # Combined: top-k from movie + top-k from book, then pick overall top-k
        combined = torch.cat([movie_scores, book_scores])
        _, topk = torch.topk(combined, k)
    return topk.tolist()

# ============================================================
# 6. Evaluation & Recommendations
# ============================================================
print("\n" + "=" * 60)
print("  Model Inference & Recommendation Results")
print("=" * 60)

# 6.1 Show recommendations for sample users
sample_users = sorted(test_seqs.keys())[:5]
for uid in sample_users:
    ts = test_seqs[uid]
    last_item = ts["sxy"][-1][1] if ts["sxy"] else None
    last_dom = "Movie" if ts["sxy"] and ts["sxy"][-1][2] == 0 else "Book"

    print(f"\n{'-'*60}")
    hist_len = len(train_seqs.get(uid,{}).get('sxy',[]))
    print(f"User {uid} | History: {hist_len} interactions | Ground-truth: {get_item_name(last_item) if last_item else 'N/A'} ({last_dom})")

    recs = recommend(uid, k=5, domain="all")
    if recs:
        hit = "HIT!" if last_item in recs else "miss"
        print(f"Combined Top-5 ({hit}):")
        for r, idx in enumerate(recs):
            marker = " <-- HIT!" if idx == last_item else ""
            print(f"  {r+1}. {get_item_name(idx)}{marker}")
        # Per-domain
        recs_movie = recommend(uid, k=3, domain="movie")
        if recs_movie:
            names = [get_item_name(i)[:50] for i in recs_movie]
            print(f"Movie Top-3: {' | '.join(names)}")
        recs_book = recommend(uid, k=3, domain="book")
        if recs_book:
            names = [get_item_name(i)[:50] for i in recs_book]
            print(f"Book Top-3:  {' | '.join(names)}")

# 6.2 Full test evaluation
print(f"\n{'='*60}")
print("  Test Set Evaluation (HR@10 / NDCG@10 / MRR)")
print("=" * 60)

hits, ndcg_sum, mrr_sum, total = 0, 0, 0, 0
dh, dt = {0:0, 1:0}, {0:0, 1:0}
n_eval = min(2000, len(test_seqs))

for uid in tqdm(list(test_seqs.keys())[:n_eval], desc="Evaluating"):
    ts = test_seqs[uid]
    if not ts["sxy"]:
        continue
    target = ts["sxy"][-1][1]
    dom = ts["sxy"][-1][2]

    movie_scores, book_scores, _ = get_scores(uid)
    if movie_scores is None:
        continue

    # Use domain-specific scores for evaluation (matches training eval protocol)
    scores = movie_scores if dom == 0 else book_scores
    _, topk = torch.topk(scores, 10)
    topk = topk.tolist()

    if target in topk:
        rank = topk.index(target) + 1
        hits += 1; dh[dom] += 1
        ndcg_sum += 1.0 / math.log2(rank + 1)
        mrr_sum += 1.0 / rank
    total += 1; dt[dom] += 1

hr = hits / total if total else 0
ndcg = ndcg_sum / total if total else 0
mrr = mrr_sum / total if total else 0

print(f"\n{'='*50}")
print(f"  Final Results")
print(f"{'='*50}")
print(f"  Evaluated samples: {total}")
print(f"  HR@10:   {hr:.4f}")
print(f"  NDCG@10: {ndcg:.4f}")
print(f"  MRR:     {mrr:.4f}")
for dom, name in [(0, "Movie"), (1, "Book")]:
    if dt[dom] > 0:
        print(f"  {name} HR@10: {dh[dom]/dt[dom]:.4f} ({dh[dom]}/{dt[dom]})")

print(f"\n{'='*60}")
print("  Done!")
print("=" * 60)
