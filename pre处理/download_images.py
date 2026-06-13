"""
图片批量下载
读取 final/item_meta_merged.csv → 多线程下载 imUrl → 保存到 final/images/
"""
import os, json, requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 配置
FINAL_DIR = "./final"
META_CSV = f"{FINAL_DIR}/item_meta_merged.csv"
IMAGE_DIR = f"{FINAL_DIR}/images"
MAPPING_FILE = f"{FINAL_DIR}/item_image_map.json"
NUM_THREADS = 64
TIMEOUT = 10
os.makedirs(IMAGE_DIR, exist_ok=True)

# 加载元数据
meta_df = pd.read_csv(META_CSV)
print(f"总物品: {len(meta_df)}")
print(f"有URL: {meta_df['imUrl'].notna().sum()}")

urls = meta_df[['parent_asin', 'imUrl']].dropna(subset=['imUrl'])
urls = urls[urls['imUrl'].str.startswith('http')]
print(f"有效URL: {len(urls)}")


def download_image(row):
    item_id, url = row['parent_asin'], row['imUrl']
    img_path = os.path.join(IMAGE_DIR, f"{item_id}.jpg")

    if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
        return item_id, img_path, True

    try:
        resp = requests.get(url, timeout=TIMEOUT, stream=True)
        if resp.status_code == 200:
            with open(img_path, 'wb') as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            if os.path.getsize(img_path) > 0:
                return item_id, img_path, True
    except Exception:
        pass

    if os.path.exists(img_path):
        os.remove(img_path)
    return item_id, None, False


results = {}
failed_items = set()

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = {executor.submit(download_image, row): row for _, row in urls.iterrows()}
    for future in tqdm(as_completed(futures), total=len(futures), desc="下载图片"):
        item_id, path, ok = future.result()
        if ok:
            results[str(item_id)] = path
        else:
            failed_items.add(str(item_id))

with open(MAPPING_FILE, 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"下载成功: {len(results)} 张, 失败: {len(failed_items)} 张")
