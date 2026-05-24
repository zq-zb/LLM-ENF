import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import pandas as pd
import numpy as np
import os
import json
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from config import (
    ITEM_META_LLM, ITEM_IMAGE_MAP_PATH, IMAGE_DIR, 
    FEATURE_SAVE_DIR, CLIP_MODEL, MODAL_DIM
)

class ImageDataset(Dataset):
    def __init__(self, df, img_path_dict):
        self.item_ids = df["item_id"].tolist()
        self.img_paths = [img_path_dict.get(str(i), None) for i in self.item_ids]
    
    def __len__(self):
        return len(self.item_ids)
    
    def __getitem__(self, idx):
        item_id = self.item_ids[idx]
        img_path = self.img_paths[idx]
        return item_id, img_path
    
def collate_fn(batch):
    item_ids = [b[0] for b in batch]
    img_paths = [b[1] for b in batch]
    return item_ids, img_paths

def extract_image_features():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    
    # 加载模型
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    torch.set_grad_enabled(False)
    
    # 加载元数据
    df = pd.read_csv(ITEM_META_LLM)
    print(f"加载数据，商品数：{len(df)}")
    
    # 预先构建所有图片路径字典
    img_map = {}
    if os.path.exists(ITEM_IMAGE_MAP_PATH):
        with open(ITEM_IMAGE_MAP_PATH, "r", encoding="utf-8") as f:
            img_map = json.load(f)
        print(f"加载 item_image_map.json，映射数：{len(img_map)}")
    
    # 补充按 item_id 命名的图片路径
    img_path_dict = {}
    for item_id in df["item_id"]:
        key = str(item_id)
        if key in img_map:
            img_path_dict[key] = img_map[key]
        else:
            # 查找常见扩展名
            found = None
            for fmt in [".jpg", ".png", ".jpeg"]:
                temp_path = os.path.join(IMAGE_DIR, f"{item_id}{fmt}")
                if os.path.exists(temp_path):
                    found = temp_path
                    break
            img_path_dict[key] = found
    
    dataset = ImageDataset(df, img_path_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=4)
    
    all_feats = []
    all_ids = []
    fail_count = 0
    
    rng = np.random.RandomState(42)
    
    for batch_ids, batch_paths in tqdm(dataloader, desc="Extracting image features"):
        # 准备当前 batch 的有效图片
        images = []
        valid_indices = []
        for i, path in enumerate(batch_paths):
            if path and os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    valid_indices.append(i)
                except:
                    pass
        
        # 特征矩阵（batch_size, MODAL_DIM），全部初始化为随机小值
        batch_feats = rng.uniform(-0.01, 0.01, (len(batch_ids), MODAL_DIM)).astype(np.float32)
        
        if images:
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                valid_feats = model.get_image_features(**inputs).cpu().numpy()
            # 填入有效位置
            for idx, feat in zip(valid_indices, valid_feats):
                batch_feats[idx] = feat
        else:
            fail_count += len(batch_ids)
        
        all_feats.append(batch_feats)
        all_ids.extend(batch_ids)
    
    all_feats = np.concatenate(all_feats, axis=0)
    np.save(f"{FEATURE_SAVE_DIR}/image_features_512.npy", all_feats)
    pd.DataFrame({"item_id": all_ids}).to_csv(f"{FEATURE_SAVE_DIR}/image_id_map.csv", index=False)
    
    print(f"成功图片数：{len(all_ids) - fail_count}，失败/填充随机数：{fail_count}")
    print(f"特征shape：{all_feats.shape}（商品数 × {MODAL_DIM}维）")

if __name__ == "__main__":
    extract_image_features()