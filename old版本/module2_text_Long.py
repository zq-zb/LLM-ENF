import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from config import ITEM_META_LLM, FEATURE_SAVE_DIR, LONG_CLIP_MODEL, MODAL_DIM

class TextDataset(Dataset):
    def __init__(self, df, text_col):
        self.texts = df[text_col].tolist()
        self.item_ids = df["item_id"].tolist()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.item_ids[idx]

def extract_batch(model, processor, dataloader, max_length, device, desc):
    model.eval()
    features = []
    ids = []
    with torch.no_grad():
        for batch_texts, batch_ids in tqdm(dataloader, desc=desc):
            inputs = processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            batch_feats = model.get_text_features(**inputs).cpu().numpy()
            features.append(batch_feats)
            ids.extend(batch_ids)
    return np.concatenate(features, axis=0), ids

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64  
    
    # 加载模型
    print(f"加载 Long-CLIP 模型：{LONG_CLIP_MODEL}")
    model = CLIPModel.from_pretrained(LONG_CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(LONG_CLIP_MODEL)
    
    # 加载数据
    df = pd.read_csv(ITEM_META_LLM)
    print(f"加载数据，商品数：{len(df)}")
    df["concat_text"] = (
        df["title"] + " | " + 
        df["llm_refined"].fillna("") + " | " + 
        df["description"].fillna("") + " | " + 
        df["llm_enhanced_text"].fillna("")
    )
    
    dataset = TextDataset(df, "concat_text")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 提取原始 Long-CLIP 特征（max_length=248）
    feats_raw, item_ids = extract_batch(model, processor, dataloader, 248, device, "Long-CLIP (248 tokens)")
    original_dim = feats_raw.shape[1]
    print(f"原始 Long-CLIP 特征维度: {original_dim}")
    
    # 维度不等于 512，则 PCA 降维
    if original_dim != MODAL_DIM:
        print(f"维度不一致，采样训练 PCA 投影到 {MODAL_DIM} 维")
        # 采样 2000 条
        sample_size = min(2000, len(feats_raw))
        sample_indices = np.random.choice(len(feats_raw), sample_size, replace=False)
        sample_feats = feats_raw[sample_indices]
        pca = PCA(n_components=MODAL_DIM, random_state=42)
        pca.fit(sample_feats)
        # 投影全部数据
        feats_aligned = pca.transform(feats_raw)
        # 保存 PCA 模型
        np.savez(f"{FEATURE_SAVE_DIR}/longclip_pca_projection.npz",
                 components=pca.components_, mean=pca.mean_)
        print("PCA 投影完成，特征已对齐到 512 维")
    else:
        feats_aligned = feats_raw
        print("维度已是 512，无需降维")
    
    # 保存最终特征
    np.save(f"{FEATURE_SAVE_DIR}/text_features_longclip_aligned_512.npy", feats_aligned)
    pd.DataFrame({"item_id": item_ids}).to_csv(f"{FEATURE_SAVE_DIR}/text_id_map_longclip.csv", index=False)
    print(f"完成，特征 shape: {feats_aligned.shape}")

if __name__ == "__main__":
    main()