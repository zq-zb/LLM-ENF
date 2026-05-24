# 社会计算 课程大作业 — 跨域多模态推荐 (Movie-Book CDSR)

## 项目概览
本仓库实现了一个面向电影与图书（Movie ↔ Book）的跨域推荐数据处理与多模态特征提取流水线。核心流程涵盖：数据清洗、ID-协同嵌入训练、文本与图像特征抽取（CLIP / Long-CLIP）、以及基于自注意力的用户序列编码，最终生成每个用户的 9 组偏好向量（形状：[9, 512]）。

## 快速导航
- 数据目录: [movie_book_cdsr_processed](movie_book_cdsr_processed)
- 主要 Notebook: [module嵌入.ipynb](module嵌入.ipynb)，[module建模.ipynb](module建模.ipynb)
- 特征提取脚本: [module1_id_embedding.py](module1_id_embedding.py), [module2_text_CLIP.py](module2_text_CLIP.py), [module2_text_Long.py](module2_text_Long.py), [module3_image_feature.py](module3_image_feature.py)
- 辅助脚本: [csvclean.py](csvclean.py), [config.py](config.py)
- 输出示例目录: [user_preferences](user_preferences)（以及 movie_book_cdsr_processed/user_preferences2）

## Notebook 区别（重要）
- `module建模.ipynb`：汇聚三类全局嵌入（ID / 图像 / 文本），示例中使用 Long-CLIP 文本特征（`text_features_longclip_512.npy` / `text_features_longclip_aligned_512.npy`）并将用户偏好保存到 `./user_preferences/`。 这里是Long-CLIP的偏好向量
- `module建模.ipynb`：结构与编码器实现基本一致，但使用 CLIP 文本特征（`text_features_clip_512.npy`），并将结果保存为 `movie_book_cdsr_processed/user_preferences2/`。

两份 notebook 方便比较不同文本特征（CLIP vs Long-CLIP）对下游偏好表示的影响。

## 脚本说明
- `config.py`: 全局路径与模型/维度配置（`FEATURE_SAVE_DIR`, `CLIP_MODEL`, `LONG_CLIP_MODEL`, `MODAL_DIM` 等）。
- `csvclean.py`: 检测并清洗物品元信息中的脏行，生成干净 CSV 与 `bad_lines_report.txt`。
- `module1_id_embedding.py`: 使用 Word2Vec 从用户交互序列训练物品 ID 的协同嵌入，保存为 `item_id_collab_512.npy`。
- `module2_text_CLIP.py`: 使用 CLIP 模型提取文本嵌入（短上下文，示例 token 限制 77），保存为 `text_features_clip_512.npy` 和 `text_id_map_clip.csv`。
- `module2_text_Long.py`: 使用 Long-CLIP 提取长文本嵌入（示例 max_length=248），若输出维度 ≠ 512 则用 PCA 对齐至 512 并保存投影参数（`longclip_pca_projection.npz`）。输出为 `text_features_longclip_aligned_512.npy`。
- `module3_image_feature.py`: 使用 CLIP 提取图像特征；若图片缺失或打开失败，脚本会在对应位置填充小随机向量以保持对齐，输出 `image_features_512.npy` 与 `image_id_map.csv`。

## 数据与输出位置
- 主数据：`movie_book_cdsr_processed/`（包含 `train.csv`, `val.csv`, `test.csv`, `item_meta*.csv`, `item2id.json`, `user2id.json` 等）。
- 多模态特征目录示例：`movie_book_cdsr_processed/multimodal_features/` 与 `.../multimodal_features_last/`，保存如下文件：
  - `image_features_512.npy`
  - `text_features_clip_512.npy`
  - `text_features_longclip_512.npy` / `text_features_longclip_aligned_512.npy`
  - `item_id_collab_512.npy`
- 用户偏好输出：
  - `user_preferences/user_9_preferences.npy`、`user_preferences/user_ids.npy`（由 `module嵌入.ipynb` 生成的默认位置）
  - `movie_book_cdsr_processed/user_preferences2/user_9_preferences.npy`、`.../user_ids.npy`（由 `module建模.ipynb` 生成的备选位置）

## 关键实现细节
- 编码器：Notebook 中采用 9 个并行编码器（索引 0~8）：
  - 0:X^ID, 1:X^视觉, 2:X^文本,
  - 3:Y^ID, 4:Y^视觉, 5:Y^文本,
  - 6:X+Y^ID, 7:X+Y^视觉, 8:X+Y^文本
  每个编码器由 `PositionalEncoding` + `TransformerLayer` 组成，最终取序列最后位置作为偏好向量。
- 输出格式：每个用户最终得到 `9 × 512` 的偏好表示，并保存为 numpy 数组以便下游建模。
