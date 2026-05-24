import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from config import ITEM_META_LLM, FEATURE_SAVE_DIR, CLIP_MODEL, MODAL_DIM

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
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    
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
    
    # 提取特征
    feats, item_ids = extract_batch(model, processor, dataloader, 77, device, "CLIP (77 tokens)")
    
    # 保存
    np.save(f"{FEATURE_SAVE_DIR}/text_features_clip_512.npy", feats)
    pd.DataFrame({"item_id": item_ids}).to_csv(f"{FEATURE_SAVE_DIR}/text_id_map_clip.csv", index=False)
    print(f"完成，特征 shape: {feats.shape}")

if __name__ == "__main__":
    main()