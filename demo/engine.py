import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = r"D:\CR\社会计算\2\LLM-ENF"
FINAL_DIR = os.path.join(BASE_DIR, "final")
MODEL_PATH = os.path.join(BASE_DIR, "best_val_model.pt")
FEAT_DIR = os.path.join(FINAL_DIR, "features")

sys.path.insert(0, os.path.join(BASE_DIR, "训练", "主实验训练"))
from cdsr_model import CDSRModel

DEVICE = "cpu"
MAX_SEQ = 200
LAMBDA1, LAMBDA2 = 0.3, 0.1

class RecEngine:
    def __init__(self):
        print("Initializing RecEngine...")
        # 1. Load mappings & metadata
        with open(os.path.join(FINAL_DIR, "item2id.json")) as f:
            self.item2id = json.load(f)
        self.id2item = {int(v): k for k, v in self.item2id.items()}
        self.n_items = len(self.item2id)

        with open(os.path.join(FINAL_DIR, "user2id.json")) as f:
            self.user2id = json.load(f)
        self.id2user = {int(v): k for k, v in self.user2id.items()}

        self.meta = pd.read_csv(os.path.join(FINAL_DIR, "item_meta_merged.csv"))
        
        # Load item image maps
        self.item_image_map = {}
        img_map_path = os.path.join(FINAL_DIR, "item_image_map.json")
        if os.path.exists(img_map_path):
            with open(img_map_path) as f:
                self.item_image_map = json.load(f)

        # Domain map
        self.domain_map = {}
        self.item_names = {}
        self.descriptions = {}
        for _, row in self.meta.iterrows():
            idx = int(row["parent_asin"])
            self.domain_map[idx] = int(row["domain"])
            self.item_names[idx] = str(row.get("title", ""))
            self.descriptions[idx] = str(row.get("description", ""))

        # 2. Load features
        print("Loading precomputed features...")
        self.id_feats = torch.from_numpy(np.load(os.path.join(FEAT_DIR, "item_id_lightgcn_512.npy"))).float()
        self.img_feats = torch.from_numpy(np.load(os.path.join(FEAT_DIR, "image_features_768.npy"))).float()
        self.tex_feats = torch.from_numpy(np.load(os.path.join(FEAT_DIR, "text_bge_llmdesc_1024.npy"))).float()

        # Domain masks
        self.movie_mask = torch.zeros(self.n_items, dtype=torch.bool)
        self.book_mask = torch.zeros(self.n_items, dtype=torch.bool)
        movie_idx = self.meta[self.meta["domain"] == 0]["parent_asin"].astype(int).tolist()
        book_idx = self.meta[self.meta["domain"] == 1]["parent_asin"].astype(int).tolist()
        self.movie_mask[movie_idx] = True
        self.book_mask[book_idx] = True

        # 3. Load model
        print("Loading PyTorch CDSR Model...")
        self.model = CDSRModel(
            self.n_items, self.id_feats, self.img_feats, self.tex_feats,
            self.movie_mask, self.book_mask,
            d_model=256, n_heads=4, n_layers=2, dropout=0.2, max_len=200,
            ablation="full"
        )
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

        # 4. Load interaction sequences (train + val for active profile building)
        print("Loading train, val and test sets...")
        self.train_df = pd.read_csv(os.path.join(FINAL_DIR, "train.csv"))
        self.val_df = pd.read_csv(os.path.join(FINAL_DIR, "val.csv"))
        self.test_df = pd.read_csv(os.path.join(FINAL_DIR, "test.csv"))
        self.all_interactions = pd.concat([self.train_df, self.val_df, self.test_df]).sort_values("timestamp")

        print("Building user sequences...")
        self.user_seqs = {}
        for uid, g in self.all_interactions.groupby("user_id"):
            uid = int(uid)
            sx, sy, sxy = [], [], []
            for _, row in g.iterrows():
                ts = row["timestamp"]
                iid = int(row["item_id"])
                dom = self.domain_map.get(iid, 1)
                if dom == 0:
                    sx.append((ts, iid))
                else:
                    sy.append((ts, iid))
                sxy.append((ts, iid, dom))
            self.user_seqs[uid] = {"sx": sx, "sy": sy, "sxy": sxy}

        # Select Top 50 Active Users
        user_counts = self.all_interactions["user_id"].value_counts()
        self.top_50_users = user_counts.head(50).index.tolist()
        print("RecEngine initialization complete.")

    def get_item_info(self, item_id):
        item_id = int(item_id)
        img_path = self.item_image_map.get(str(item_id), "")
        # convert local path for routing if needed (e.g. final/images/...)
        if img_path:
            # normalize paths like './final/images\\41.jpg' to web-friendly paths
            img_path = img_path.replace("\\", "/").replace("./final/", "/static/")
        else:
            img_path = "/static/images/default.jpg"

        return {
            "id": item_id,
            "title": self.item_names.get(item_id, f"Item {item_id}"),
            "description": self.descriptions.get(item_id, ""),
            "domain": "movie" if self.domain_map.get(item_id, 1) == 0 else "book",
            "image": img_path
        }

    def search_items(self, query, domain="all", limit=30):
        query = str(query).strip().lower()
        if not query:
            return []
        
        matches = []
        for idx, title in self.item_names.items():
            if query in title.lower():
                dom = "movie" if self.domain_map.get(idx, 1) == 0 else "book"
                if domain != "all" and dom != domain:
                    continue
                matches.append(self.get_item_info(idx))
                if len(matches) >= limit:
                    break
        return matches

    def get_popular(self, domain="movie", limit=20):
        dom_val = 0 if domain == "movie" else 1
        subset = self.all_interactions[self.all_interactions["domain"] == dom_val]
        counts = subset["item_id"].value_counts().head(limit)
        return [self.get_item_info(iid) for iid in counts.index]

    def pad_seq(self, ids):
        ids = ids[-MAX_SEQ:]
        n = len(ids)
        return (torch.tensor(ids + [0]*(MAX_SEQ-n), dtype=torch.long).unsqueeze(0),
                torch.tensor([1]*n + [0]*(MAX_SEQ-n), dtype=torch.long).unsqueeze(0))

    @torch.no_grad()
    def get_recommendations_for_user(self, user_id, target_domain="book", top_k=10):
        user_id = int(user_id)
        ctx = self.user_seqs.get(user_id, {"sx": [], "sy": [], "sxy": []})
        sxy_ids = [z[1] for z in ctx["sxy"]]
        if not sxy_ids:
            return []

        sx_ids = [x[1] for x in ctx["sx"]]
        sy_ids = [y[1] for y in ctx["sy"]]

        sx_t, sx_m = self.pad_seq(sx_ids)
        sy_t, sy_m = self.pad_seq(sy_ids)
        sxy_t, sxy_m = self.pad_seq(sxy_ids)

        out = self.model(sx_t, sx_m, sy_t, sy_m, sxy_t, sxy_m, return_cross=True)

        if target_domain == "movie":
            scores = out["P_X"][0] + LAMBDA1*out["P_Y_to_X"][0] + LAMBDA2*out["P_XY_to_X"][0]
        else:
            scores = out["P_X_to_Y"][0] + LAMBDA1*out["P_Y"][0] + LAMBDA2*out["P_XY_to_Y"][0]

        # Filter out interacted items
        scores[sxy_ids] = float('-inf')

        # Softmax scores for visual representation
        scores_softmax = F.softmax(scores, dim=0)

        top_scores, top_indices = torch.topk(scores_softmax, top_k * 2)
        
        recs = []
        for s, idx in zip(top_scores.tolist(), top_indices.tolist()):
            # double check domain
            dom = "movie" if self.domain_map.get(idx, 1) == 0 else "book"
            if dom != target_domain:
                continue
            item_info = self.get_item_info(idx)
            item_info["score"] = float(s)
            recs.append(item_info)
            if len(recs) >= top_k:
                break
        return recs

    @torch.no_grad()
    def get_recommendations_from_sequence(self, sequence_ids, target_domain="book", top_k=10):
        # Build sx, sy, sxy from a manual list of sequence IDs (for the custom recommendations mode)
        sx_ids = [iid for iid in sequence_ids if self.domain_map.get(iid, 1) == 0]
        sy_ids = [iid for iid in sequence_ids if self.domain_map.get(iid, 1) == 1]
        sxy_ids = list(sequence_ids)

        sx_t, sx_m = self.pad_seq(sx_ids)
        sy_t, sy_m = self.pad_seq(sy_ids)
        sxy_t, sxy_m = self.pad_seq(sxy_ids)

        out = self.model(sx_t, sx_m, sy_t, sy_m, sxy_t, sxy_m, return_cross=True)

        if target_domain == "movie":
            scores = out["P_X"][0] + LAMBDA1*out["P_Y_to_X"][0] + LAMBDA2*out["P_XY_to_X"][0]
        else:
            scores = out["P_X_to_Y"][0] + LAMBDA1*out["P_Y"][0] + LAMBDA2*out["P_XY_to_Y"][0]

        # Filter out interacted items
        scores[sxy_ids] = float('-inf')
        scores_softmax = F.softmax(scores, dim=0)

        top_scores, top_indices = torch.topk(scores_softmax, top_k * 2)

        recs = []
        for s, idx in zip(top_scores.tolist(), top_indices.tolist()):
            dom = "movie" if self.domain_map.get(idx, 1) == 0 else "book"
            if dom != target_domain:
                continue
            item_info = self.get_item_info(idx)
            item_info["score"] = float(s)
            recs.append(item_info)
            if len(recs) >= top_k:
                break
        return recs

    def get_user_profile(self, user_id):
        user_id = int(user_id)
        ctx = self.user_seqs.get(user_id, {"sx": [], "sy": [], "sxy": []})
        
        history = [self.get_item_info(iid) for _, iid in sorted(ctx["sxy"], key=lambda x: x[0])]
        
        movies = [item for item in history if item["domain"] == "movie"]
        books = [item for item in history if item["domain"] == "book"]
        
        return {
            "user_id": user_id,
            "username": self.id2user.get(user_id, f"User {user_id}"),
            "history_count": len(history),
            "movie_count": len(movies),
            "book_count": len(books),
            "history": history
        }
