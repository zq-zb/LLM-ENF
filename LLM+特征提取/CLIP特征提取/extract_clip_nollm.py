"""CLIP 文本特征"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch, pandas as pd, numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

df = pd.read_csv("final/item_meta_llm_enhanced.csv")
# 纯原始文本，无 LLM
df["clip_text_raw"] = (
    df["title"].fillna("") + " | " + df["description"].fillna("").str.slice(0, 200)
)
print(f"Items: {len(df)}  avg text len: {df['clip_text_raw'].str.len().mean():.0f}")

class TextDataset(Dataset):
    def __init__(self, texts, item_ids):
        self.texts = texts; self.ids = item_ids
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx], self.ids[idx]

print("Loading CLIP ViT-B/32...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

texts = df["clip_text_raw"].tolist()
ids = df["parent_asin"].tolist()
loader = DataLoader(TextDataset(texts, ids), batch_size=64, shuffle=False, num_workers=0)

feats, all_ids = [], []
with torch.no_grad():
    for batch_texts, batch_ids in tqdm(loader, desc="CLIP raw (77)"):
        inp = proc(text=batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=77).to(device)
        bf = model.get_text_features(**inp).cpu().numpy()
        feats.append(bf)
        all_ids.extend(int(x) for x in batch_ids)

feats = np.concatenate(feats, axis=0)
np.save("final/features/text_features_clip_raw_512.npy", feats)
pd.DataFrame({"item_id": all_ids}).to_csv("final/features/text_id_map_clip_raw.csv", index=False)
print(f"Done: {feats.shape}")
