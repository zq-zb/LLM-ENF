"""
CDSR 训练脚本 — cloud4 版（图文对比对齐 Loss）

与 cdsr_cloud2/train.ipynb 的唯一差异：训练时新增图文对比对齐 loss，
测试"特征对齐是否是图像模块零增益的根因"。

用法：python train.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math, json, datetime
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gc, warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

from cdsr_model import CDSRModel

# ==================== 配置 ====================
TEXT_VARIANT = "llmdesc"  # desc / llm / llmdesc

DATA_DIR = "final"
TEXT_FEAT_MAP = {
    "desc":    "text_bge_desc_1024.npy",
    "llm":     "text_bge_llm_1024.npy",
    "llmdesc": "text_bge_llmdesc_1024.npy",
}
TEXT_FEAT_PATH = os.path.join(DATA_DIR, "features", TEXT_FEAT_MAP[TEXT_VARIANT])
ID_FEAT_PATH   = os.path.join(DATA_DIR, "features", "item_id_lightgcn_512.npy")
IMG_FEAT_PATH  = os.path.join(DATA_DIR, "features", "image_features_768.npy")
TRAIN_CSV      = os.path.join(DATA_DIR, "train.csv")
VAL_CSV        = os.path.join(DATA_DIR, "val.csv")
TEST_CSV       = os.path.join(DATA_DIR, "test.csv")
ITEM2ID_PATH   = os.path.join(DATA_DIR, "item2id.json")
META_CSV       = os.path.join(DATA_DIR, "item_meta_merged.csv")

SAVE_DIR = os.path.join(DATA_DIR, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = os.path.join(DATA_DIR, "results_align.txt")

# 超参数
D_MODEL      = 256
N_HEADS      = 4
N_LAYERS     = 2
DROPOUT      = 0.2
MAX_SEQ_LEN  = 200
BATCH_SIZE   = 256
LR           = 1.4e-3
EPOCHS       = 10
PATIENCE     = 5
LAMBDA1      = 0.3
LAMBDA2      = 0.1
LABEL_SMOOTH = 0.0
USE_AMP      = True
SAVE_EVERY   = 2000
RESUME       = True
NUM_WORKERS  = 16
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# 图文对齐 Loss 超参数
ALIGN_LAMBDA  = 0.1   # 对齐 loss 权重（调参空间：0.01 / 0.05 / 0.1 / 0.5）


def log(msg=""):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


log("=" * 60)
log(f"CDSR 训练 (cloud4, +图文对齐loss) — "
    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Text: {TEXT_VARIANT}  ALIGN_LAMBDA={ALIGN_LAMBDA}")
log(f"Device: {DEVICE}  AMP={USE_AMP}  BS={BATCH_SIZE}  Epochs={EPOCHS}")
log("=" * 60)

# ==================== 加载特征 ====================
log("\n=== 加载特征 ===")
id_feats = torch.from_numpy(np.load(ID_FEAT_PATH)).float()
img_feats = torch.from_numpy(np.load(IMG_FEAT_PATH)).float()
tex_feats = torch.from_numpy(np.load(TEXT_FEAT_PATH)).float()
log(f"ID:    {id_feats.shape}  (LightGCN)")
log(f"Image: {img_feats.shape}  (ViT-L/14)")
log(f"Text:  {tex_feats.shape}  (BGE-large, {TEXT_VARIANT})")

with open(ITEM2ID_PATH) as f:
    item2id = json.load(f)
N_ITEMS = len(item2id)
log(f"物品总数: {N_ITEMS}")
assert id_feats.shape[0] == img_feats.shape[0] == tex_feats.shape[0] == N_ITEMS

meta_df = pd.read_csv(META_CSV)
movie_mask = torch.zeros(N_ITEMS, dtype=torch.bool)
book_mask  = torch.zeros(N_ITEMS, dtype=torch.bool)
movie_ids = meta_df[meta_df['domain'] == 0].index.tolist()
book_ids  = meta_df[meta_df['domain'] == 1].index.tolist()
movie_mask[movie_ids] = True
book_mask[book_ids]  = True
log(f"电影: {movie_mask.sum().item()}  图书: {book_mask.sum().item()}")

# ==================== 构建序列 & 样本 ====================
def build_user_sequences(df):
    user_seqs = {}
    for uid, g in df.groupby("user_id"):
        g = g.sort_values("timestamp")
        uid = int(uid)
        sx, sy, sxy = [], [], []
        for _, row in g.iterrows():
            ts = row["timestamp"]
            iid = int(row["item_id"])
            dom = int(row["domain"])
            if dom == 0:
                sx.append((ts, iid))
            else:
                sy.append((ts, iid))
            sxy.append((ts, iid, dom))
        user_seqs[uid] = {"sx": sx, "sy": sy, "sxy": sxy}
    return user_seqs


def build_samples(user_seqs):
    samples = []
    for uid, seqs in user_seqs.items():
        sx, sy, sxy = seqs["sx"], seqs["sy"], seqs["sxy"]
        if len(sxy) < 2:
            continue
        sx_ptr = sy_ptr = 0
        for k in range(1, len(sxy)):
            ts_k = sxy[k][0]
            while sx_ptr < len(sx) and sx[sx_ptr][0] < ts_k:
                sx_ptr += 1
            while sy_ptr < len(sy) and sy[sy_ptr][0] < ts_k:
                sy_ptr += 1
            samples.append(dict(sx_ids=[x[1] for x in sx[:sx_ptr]],
                                sy_ids=[y[1] for y in sy[:sy_ptr]],
                                sxy_ids=[z[1] for z in sxy[:k]],
                                target=sxy[k][1], target_domain=sxy[k][2],
                                seq_type=2))
        sx_to_sxy, sx_cnt = {}, 0
        for p, (ts, iid, dom) in enumerate(sxy):
            if dom == 0:
                sx_to_sxy[sx_cnt] = p; sx_cnt += 1
        sy_ptr = 0
        for i in range(1, len(sx)):
            while sy_ptr < len(sy) and sy[sy_ptr][0] < sx[i][0]:
                sy_ptr += 1
            samples.append(dict(sx_ids=[x[1] for x in sx[:i]],
                                sy_ids=[y[1] for y in sy[:sy_ptr]],
                                sxy_ids=[z[1] for z in sxy[:sx_to_sxy[i]]],
                                target=sx[i][1], target_domain=0, seq_type=0))
        sy_to_sxy, sy_cnt = {}, 0
        for p, (ts, iid, dom) in enumerate(sxy):
            if dom == 1:
                sy_to_sxy[sy_cnt] = p; sy_cnt += 1
        sx_ptr = 0
        for j in range(1, len(sy)):
            while sx_ptr < len(sx) and sx[sx_ptr][0] < sy[j][0]:
                sx_ptr += 1
            samples.append(dict(sx_ids=[x[1] for x in sx[:sx_ptr]],
                                sy_ids=[y[1] for y in sy[:j]],
                                sxy_ids=[z[1] for z in sxy[:sy_to_sxy[j]]],
                                target=sy[j][1], target_domain=1, seq_type=1))
    return samples


def build_eval_samples(context_seqs, target_seqs):
    samples = []
    for uid, t_seqs in target_seqs.items():
        if uid not in context_seqs:
            continue
        ctx = context_seqs[uid]
        sx = ctx["sx"] + t_seqs["sx"]
        sy = ctx["sy"] + t_seqs["sy"]
        sxy = ctx["sxy"] + t_seqs["sxy"]
        if len(sxy) < 2:
            continue
        k = len(sxy) - 1
        ts_k = sxy[k][0]
        sx_ptr = len([x for x in sx if x[0] < ts_k])
        sy_ptr = len([y for y in sy if y[0] < ts_k])
        samples.append(dict(sx_ids=[x[1] for x in sx[:sx_ptr]],
                            sy_ids=[y[1] for y in sy[:sy_ptr]],
                            sxy_ids=[z[1] for z in sxy[:k]],
                            target=sxy[k][1], target_domain=sxy[k][2]))
    return samples


log("\n=== 构建序列 ===")
train_seqs = build_user_sequences(pd.read_csv(TRAIN_CSV))
val_seqs   = build_user_sequences(pd.read_csv(VAL_CSV))
test_seqs  = build_user_sequences(pd.read_csv(TEST_CSV))

train_samples = build_samples(train_seqs)
val_samples = build_eval_samples(train_seqs, val_seqs)

train_val_seqs = {}
for uid in set(train_seqs.keys()) | set(val_seqs.keys()):
    ts = train_seqs.get(uid, {"sx": [], "sy": [], "sxy": []})
    vs = val_seqs.get(uid, {"sx": [], "sy": [], "sxy": []})
    train_val_seqs[uid] = {"sx": ts["sx"] + vs["sx"],
                           "sy": ts["sy"] + vs["sy"],
                           "sxy": ts["sxy"] + vs["sxy"]}
test_samples = build_eval_samples(train_val_seqs, test_seqs)

log(f"Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}")

# ==================== Dataset & DataLoader ====================
class CDSRDataset(Dataset):
    def __init__(self, samples, max_len=MAX_SEQ_LEN, is_train=True):
        self.samples = samples; self.max_len = max_len; self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        def pad(ids):
            ids = ids[-self.max_len:]
            n = len(ids)
            return torch.tensor(ids + [0] * (self.max_len - n), dtype=torch.long), \
                   torch.tensor([1] * n + [0] * (self.max_len - n), dtype=torch.long)
        sx, sx_mask   = pad(s["sx_ids"])
        sy, sy_mask   = pad(s["sy_ids"])
        sxy, sxy_mask = pad(s["sxy_ids"])
        if self.is_train:
            return (sx, sx_mask, sy, sy_mask, sxy, sxy_mask,
                    torch.tensor(s["target"], dtype=torch.long),
                    torch.tensor(s["target_domain"], dtype=torch.long),
                    torch.tensor(s["seq_type"], dtype=torch.long))
        else:
            return (sx, sx_mask, sy, sy_mask, sxy, sxy_mask,
                    torch.tensor(s["target"], dtype=torch.long),
                    torch.tensor(s["target_domain"], dtype=torch.long))


def collate_train(batch):
    sx, sx_mask, sy, sy_mask, sxy, sxy_mask, target, tgt_dom, seq_type = zip(*batch)
    return (torch.stack(list(sx)), torch.stack(list(sx_mask)),
            torch.stack(list(sy)), torch.stack(list(sy_mask)),
            torch.stack(list(sxy)), torch.stack(list(sxy_mask)),
            torch.stack(list(target)), torch.stack(list(tgt_dom)),
            torch.stack(list(seq_type)))


def collate_eval(batch):
    sx, sx_mask, sy, sy_mask, sxy, sxy_mask, target, tgt_dom = zip(*batch)
    return (torch.stack(list(sx)), torch.stack(list(sx_mask)),
            torch.stack(list(sy)), torch.stack(list(sy_mask)),
            torch.stack(list(sxy)), torch.stack(list(sxy_mask)),
            torch.stack(list(target)), torch.stack(list(tgt_dom)))


train_ds = CDSRDataset(train_samples, is_train=True)
val_ds   = CDSRDataset(val_samples,   is_train=False)
test_ds  = CDSRDataset(test_samples,  is_train=False)

NUM_WORKERS_ACTUAL = min(NUM_WORKERS, os.cpu_count() or 4)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_train, num_workers=NUM_WORKERS_ACTUAL,
                           pin_memory=True, persistent_workers=NUM_WORKERS_ACTUAL > 0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_eval, num_workers=NUM_WORKERS_ACTUAL,
                           pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_eval, num_workers=NUM_WORKERS_ACTUAL,
                           pin_memory=True)

log(f"Train batches/epoch: {len(train_loader)}  Val: {len(val_loader)}")

# ==================== 模型 ====================
log("\n=== 初始化模型 ===")
model = CDSRModel(N_ITEMS, id_feats, img_feats, tex_feats,
                   movie_mask, book_mask).to(DEVICE)
log(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

scaler = torch.cuda.amp.GradScaler() if USE_AMP and DEVICE == "cuda" else None


@torch.no_grad()
def evaluate(loader, k=10):
    model.eval()
    hits, ndcg_sum, mrr_sum, total = 0, 0, 0, 0
    dh, dt = {0: 0, 1: 0}, {0: 0, 1: 0}
    for sx, sx_mask, sy, sy_mask, sxy, sxy_mask, target, tgt_dom in loader:
        sx, sx_mask = sx.to(DEVICE), sx_mask.to(DEVICE)
        sy, sy_mask = sy.to(DEVICE), sy_mask.to(DEVICE)
        sxy, sxy_mask = sxy.to(DEVICE), sxy_mask.to(DEVICE)
        target = target.to(DEVICE); tgt_dom = tgt_dom.to(DEVICE)
        out = model(sx, sx_mask, sy, sy_mask, sxy, sxy_mask, return_cross=True)
        for i in range(len(target)):
            dom = tgt_dom[i].item()
            if dom == 0:
                score = (out["P_X"][i] + LAMBDA1 * out["P_Y_to_X"][i]
                         + LAMBDA2 * out["P_XY_to_X"][i])
            else:
                score = (out["P_X_to_Y"][i] + LAMBDA1 * out["P_Y"][i]
                         + LAMBDA2 * out["P_XY_to_Y"][i])
            _, topk = torch.topk(score, k)
            topk = topk.cpu().tolist()
            tgt = target[i].item()
            if tgt in topk:
                rank = topk.index(tgt) + 1
                hits += 1; dh[dom] += 1
                ndcg_sum += 1.0 / math.log2(rank + 1)
                mrr_sum += 1.0 / rank
            total += 1; dt[dom] += 1
        del out
    hr = hits / total if total else 0
    ndcg = ndcg_sum / total if total else 0
    mrr = mrr_sum / total if total else 0
    return hr, ndcg, mrr, total, dh, dt


# ==================== 训练循环 ====================
log("\n=== 开始训练 ===")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01,
                               fused=True if DEVICE == "cuda" else False)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

start_epoch = 1
best_val_hr = 0.0
best_val_mrr = 0.0
no_improve = 0
ckpt_path = os.path.join(SAVE_DIR, "checkpoint_align.pt")

if RESUME and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_val_hr = ckpt.get("best_val_hr", 0.0)
    best_val_mrr = ckpt.get("best_val_mrr", 0.0)
    no_improve = ckpt["no_improve"]
    log(f"从断点恢复: epoch={start_epoch}, best_val_hr={best_val_hr:.4f}")
else:
    log("从头开始训练")

log(f"{'Epoch':>6} {'Loss':>8} {'Main':>8} {'Align':>8} "
    f"{'ValHR':>8} {'ValNDCG':>8} {'ValMRR':>8} "
    f"{'TestHR':>8} {'TestNDCG':>8} {'TestMRR':>8} {'MovieHR':>8} {'BookHR':>8}")
log("-" * 100)

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss, total_main, total_align, n_batch = 0, 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for sx, sx_mask, sy, sy_mask, sxy, sxy_mask, target, tgt_dom, seq_type in pbar:
        sx, sx_mask = sx.to(DEVICE), sx_mask.to(DEVICE)
        sy, sy_mask = sy.to(DEVICE), sy_mask.to(DEVICE)
        sxy, sxy_mask = sxy.to(DEVICE), sxy_mask.to(DEVICE)
        target, seq_type = target.to(DEVICE), seq_type.to(DEVICE)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(sx, sx_mask, sy, sy_mask, sxy, sxy_mask,
                            return_cross=False)
            main_loss = torch.tensor(0.0, device=DEVICE)
            for st, wt, key in [(0, 1.0, "P_X"), (1, LAMBDA1, "P_Y"),
                                 (2, LAMBDA2, "P_XY")]:
                m = (seq_type == st)
                if m.any():
                    main_loss = main_loss + wt * criterion(out[key][m], target[m])

            # ── 图文对齐 loss ──
            all_ids = torch.cat([sx.flatten(), sy.flatten(), sxy.flatten()], dim=0)
            align_loss = model.compute_alignment_loss(all_ids)
            loss = main_loss + ALIGN_LAMBDA * align_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(sx, sx_mask, sy, sy_mask, sxy, sxy_mask,
                        return_cross=False)
            main_loss = torch.tensor(0.0, device=DEVICE)
            for st, wt, key in [(0, 1.0, "P_X"), (1, LAMBDA1, "P_Y"),
                                 (2, LAMBDA2, "P_XY")]:
                m = (seq_type == st)
                if m.any():
                    main_loss = main_loss + wt * criterion(out[key][m], target[m])

            all_ids = torch.cat([sx.flatten(), sy.flatten(), sxy.flatten()], dim=0)
            align_loss = model.compute_alignment_loss(all_ids)
            loss = main_loss + ALIGN_LAMBDA * align_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_main += main_loss.item()
        total_align += align_loss.item()
        n_batch += 1
        pbar.set_postfix({"loss": f"{loss.item():.3f}",
                          "align": f"{align_loss.item():.3f}"})

        if n_batch % SAVE_EVERY == 0:
            torch.save(dict(
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict() if scaler else {},
                epoch=epoch, best_val_hr=best_val_hr, best_val_mrr=best_val_mrr,
                no_improve=no_improve), ckpt_path)

    avg_loss = total_loss / n_batch
    avg_main = total_main / n_batch
    avg_align = total_align / n_batch

    gc.collect(); torch.cuda.empty_cache()
    val_hr, val_ndcg, val_mrr, val_n, val_dh, val_dt = evaluate(val_loader)
    gc.collect(); torch.cuda.empty_cache()
    test_hr, test_ndcg, test_mrr, test_n, test_dh, test_dt = evaluate(test_loader)
    movie_hr = test_dh[0] / test_dt[0] if test_dt[0] else 0
    book_hr = test_dh[1] / test_dt[1] if test_dt[1] else 0

    log(f"{epoch:>6} {avg_loss:>8.4f} {avg_main:>8.4f} {avg_align:>8.4f} "
        f"{val_hr:>8.4f} {val_ndcg:>8.4f} {val_mrr:>8.4f} "
        f"{test_hr:>8.4f} {test_ndcg:>8.4f} {test_mrr:>8.4f} "
        f"{movie_hr:>8.4f} {book_hr:>8.4f}")

    scheduler.step(val_hr)

    if val_hr > best_val_hr + 1e-5:
        best_val_hr = val_hr; no_improve = 0
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "best_model_align.pt"))
        log("  -> 已保存 (best hr)")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            log(f"早停于 epoch {epoch}"); break

    if val_mrr > best_val_mrr:
        best_val_mrr = val_mrr
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "best_val_model_align.pt"))

    torch.save(dict(
        model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict() if scaler else {},
        epoch=epoch, best_val_hr=best_val_hr, best_val_mrr=best_val_mrr,
        no_improve=no_improve), ckpt_path)

log(f"\n训练结束. Best Val HR@10: {best_val_hr:.4f}  "
    f"Best Val MRR: {best_val_mrr:.4f}")

# ==================== 最终评估 ====================
log("\n=== 最终评估 (Best HR Model) ===")
model.load_state_dict(torch.load(
    os.path.join(SAVE_DIR, "best_model_align.pt"), map_location=DEVICE))
model.eval()


@torch.no_grad()
def evaluate_final(loader, k=10):
    hits, ndcg_sum, mrr_sum, total = 0, 0, 0, 0
    dh, dt = {0: 0, 1: 0}, {0: 0, 1: 0}
    for sx, sx_mask, sy, sy_mask, sxy, sxy_mask, target, tgt_dom in tqdm(
            loader, desc="Eval"):
        sx, sx_mask = sx.to(DEVICE), sx_mask.to(DEVICE)
        sy, sy_mask = sy.to(DEVICE), sy_mask.to(DEVICE)
        sxy, sxy_mask = sxy.to(DEVICE), sxy_mask.to(DEVICE)
        target = target.to(DEVICE); tgt_dom = tgt_dom.to(DEVICE)
        out = model(sx, sx_mask, sy, sy_mask, sxy, sxy_mask, return_cross=True)
        for i in range(len(target)):
            dom = tgt_dom[i].item()
            if dom == 0:
                score = (out["P_X"][i] + LAMBDA1 * out["P_Y_to_X"][i]
                         + LAMBDA2 * out["P_XY_to_X"][i])
            else:
                score = (out["P_X_to_Y"][i] + LAMBDA1 * out["P_Y"][i]
                         + LAMBDA2 * out["P_XY_to_Y"][i])
            _, topk = torch.topk(score, k)
            topk = topk.cpu().tolist()
            tgt = target[i].item()
            if tgt in topk:
                rank = topk.index(tgt) + 1
                hits += 1; dh[dom] += 1
                ndcg_sum += 1.0 / math.log2(rank + 1)
                mrr_sum += 1.0 / rank
            total += 1; dt[dom] += 1
        del out
    hr = hits / total if total else 0
    ndcg = ndcg_sum / total if total else 0
    mrr = mrr_sum / total if total else 0
    return hr, ndcg, mrr, total, dh, dt


log("--- Val ---")
val_hr, val_ndcg, val_mrr, val_n, val_dh, val_dt = evaluate_final(val_loader)
log(f"Overall: HR@10={val_hr:.4f}  NDCG@10={val_ndcg:.4f}  "
    f"MRR={val_mrr:.4f}  samples={val_n}")
for dom, name in [(0, "Movie"), (1, "Book")]:
    if val_dt[dom] > 0:
        log(f"  {name}: HR@10={val_dh[dom]/val_dt[dom]:.4f}")

log("\n--- Test ---")
test_hr, test_ndcg, test_mrr, test_n, test_dh, test_dt = evaluate_final(test_loader)
log(f"Overall: HR@10={test_hr:.4f}  NDCG@10={test_ndcg:.4f}  "
    f"MRR={test_mrr:.4f}  samples={test_n}")
for dom, name in [(0, "Movie"), (1, "Book")]:
    if test_dt[dom] > 0:
        log(f"  {name}: HR@10={test_dh[dom]/test_dt[dom]:.4f}")

# ==================== 推理时跨域消融 ====================
log("\n=== 跨域消融 (Test) ===")


def eval_ablation(loader, mode="full", k=10):
    hits, ndcg_sum, total = 0, 0, 0
    with torch.no_grad():
        for sx, sx_mask, sy, sy_mask, sxy, sxy_mask, target, tgt_dom in tqdm(
                loader, desc=f"Abl-{mode}"):
            sx, sx_mask = sx.to(DEVICE), sx_mask.to(DEVICE)
            sy, sy_mask = sy.to(DEVICE), sy_mask.to(DEVICE)
            sxy, sxy_mask = sxy.to(DEVICE), sxy_mask.to(DEVICE)
            target = target.to(DEVICE); tgt_dom = tgt_dom.to(DEVICE)
            out = model(sx, sx_mask, sy, sy_mask, sxy, sxy_mask,
                        return_cross=True)
            for i in range(len(target)):
                dom = tgt_dom[i].item()
                if dom == 0:
                    if mode == "full":
                        score = (out["P_X"][i] + LAMBDA1*out["P_Y_to_X"][i]
                                 + LAMBDA2*out["P_XY_to_X"][i])
                    elif mode == "in_only":
                        score = out["P_X"][i]
                    elif mode == "cross":
                        score = (LAMBDA1*out["P_Y_to_X"][i]
                                 + LAMBDA2*out["P_XY_to_X"][i])
                else:
                    if mode == "full":
                        score = (out["P_X_to_Y"][i] + LAMBDA1*out["P_Y"][i]
                                 + LAMBDA2*out["P_XY_to_Y"][i])
                    elif mode == "in_only":
                        score = LAMBDA1 * out["P_Y"][i]
                    elif mode == "cross":
                        score = (out["P_X_to_Y"][i]
                                 + LAMBDA2*out["P_XY_to_Y"][i])
                _, topk = torch.topk(score, k)
                topk = topk.cpu().tolist()
                tgt = target[i].item()
                if tgt in topk:
                    hits += 1
                    ndcg_sum += 1.0 / math.log2(topk.index(tgt) + 2)
                total += 1
        del out
    return hits / total if total else 0, ndcg_sum / total if total else 0


for mode in ["full", "in_only", "cross"]:
    hr, ndcg = eval_ablation(test_loader, mode=mode)
    log(f"  {mode:8s}: HR@10={hr:.4f}  NDCG@10={ndcg:.4f}")

log()
log("=" * 60)
log(f"完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)
