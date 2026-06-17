import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from config import TRAIN_CSV, ITEM2ID_PATH, FEATURE_SAVE_DIR, ID_EMBED_DIM

if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_CSV)
    sequences = []
    for user_id, group in train_df.groupby("user_id"):
        item_seq = group["item_id"].astype(str).tolist()
        sequences.append(item_seq)
    print(f"用户序列数: {len(sequences)}")

    # 训练Word2Vec
    model = Word2Vec(
        sentences=sequences,
        vector_size=ID_EMBED_DIM,
        window=7,
        min_count=1,
        sg=1,
        epochs=12,
        seed=402
    )
    print(f"模型物品总数: {len(model.wv.index_to_key)}")

    # 加载总物品数
    with open(ITEM2ID_PATH, "r") as f:
        item2id = json.load(f)
    total_items = len(item2id)

    # 生成协同矩阵
    collab_matrix = np.zeros((total_items, ID_EMBED_DIM), dtype=np.float32)
    success = 0
    for item_str in model.wv.index_to_key:
        idx = int(item_str)  # 字符串转数字 = 矩阵行号
        collab_matrix[idx] = model.wv[item_str]
        success += 1

    np.save(f"{FEATURE_SAVE_DIR}/item_id_collab_512.npy", collab_matrix)

    print(f"匹配成功: {success}/{total_items}")
    print(f"非零元素数: {np.count_nonzero(collab_matrix)}")
    print(f"第一个向量有值: {np.any(collab_matrix[0])}")
    print(f"矩阵形状: {collab_matrix.shape}")
