# LLM-EMF: 多模态跨域序列推荐

基于 LLM-EMF 框架的电影—图书跨域序列推荐系统，融合 **LLM 文本语义增强 + CLIP/BGE 多模态编码 + 分层注意力序列建模**。

## 目录结构

```
├── pre处理/                         # 数据预处理
│   ├── pre_meta_movie.ipynb               # 电影元数据清洗
│   ├── pre_meta_book.ipynb                # 图书元数据清洗
│   ├── pre_interaction_movie.ipynb        # 电影交互过滤
│   ├── pre_interaction_book.ipynb         # 图书交互过滤
│   ├── merge_and_split.ipynb              # 双域合并 + 交叉筛选 + 时序划分
│   └── download_images.py                 # 批量下载物品图片
│
├── LLM+特征提取/                    # LLM 增强 + 多模态特征提取
│   ├── vitl14提取/                        # 当前特征管线
│   │   ├── text_llm_enhance_v2.ipynb            # LLM V2 六维度语义增强 (DeepSeek)
│   │   ├── extract_id_lightgcn.ipynb            # LightGCN 512d ID 协同特征
│   │   ├── extract_image_vitl14.ipynb           # ViT-L/14 768d 图像特征
│   │   ├── extract_text_bge_3var.ipynb          # BGE-large 1024d 三变体文本特征
│   │   └── extract_id_lightgcn_v2_e2e.ipynb
│   └── CLIP特征提取/                      # 旧版 CLIP 管线（已弃用）
│       ├── text_llm_enhance.ipynb
│       ├── extract_id_embeddings.ipynb
│       ├── extract_image_features.ipynb
│       ├── extract_text_features.ipynb
│       └── extract_clip_nollm.py
│
└── 训练/                            # 模型训练与消融实验
    ├── 主实验训练/                        # 主实验（LightGCN + ViT-L/14 + BGE-large）
    │   ├── cdsr_model.py                   # 模型定义（异构特征维度，可学习跨域权重）
    │   ├── train.ipynb                     # 主训练
    │   ├── ablation.ipynb                  # 架构消融（4 变体）
    │   ├── ablation_text_variants.ipynb    # 文本变体消融（3 变体）
    │   └── requirements.txt
    ├── 主实验+图文对齐/                   # +InfoNCE 图文对齐 Loss
    │   ├── cdsr_model.py
    │   ├── train.py / train.ipynb
    │   └── requirements.txt
    └── CLIP特征训练/                      # 旧版 CLIP 统一特征训练（已弃用）
        ├── cdsr_model.py
        ├── train.ipynb
        └── results.txt
```

## 数据处理流水线

Amazon Reviews 2023 数据集，电影 (Movies_and_TV) ↔ 图书 (Books)。

| 指标 | 数值 |
|------|------:|
| 用户数 | 20,030 |
| 物品数 | 43,528（电影 21,280 + 图书 22,248） |
| 总交互 | 688,010 |
| 划分 | Train 647,950 / Val 20,030 / Test 20,030 (leave-last-2) |

### 处理步骤

1. **元数据清洗**：保留同时有 title + 有效图片 URL 的物品
2. **单域交互过滤**：迭代式双向过滤 (item>=5, user>=5)
3. **双域交叉筛选**：双域用户 -> 每域>=3 -> 总交互>=10 -> (item>=7, user>=7, 每域>=5)
4. **ID 映射 + 时序划分**：全局连续 ID，leave-last-2 策略

## 特征工程

### 多模态特征体系

| 模态 | 方法 | 维度 | 说明 |
|------|------|:---:|------|
| ID 协同 | LightGCN 3-layer | 512 | BPR 损失，weight_decay=0 |
| 图像 | CLIP ViT-L/14 | 768 | 14x14 patch，24 层 |
| 文本 | BGE-large-en-v1.5 | 1024 | 512 token，含 LLM 六维度增强 |

### LLM 文本增强

DeepSeek API，六维度结构化 Prompt (Genre / Plot / Style / Themes / Audience / Similar To)，200 词输出，20 线程并发，全量 43,528 物品。

### 文本三变体

| 变体 | 输入 | 用途 |
|------|------|------|
| A: desc | title + 原始描述[:1500] | 纯原文基线 |
| B: llm | title + LLM 生成描述 | LLM 蒸馏价值 |
| C: llmdesc | title + LLM + desc[:400] | LLM+原文互补 |

## 模型架构

**CDSRModel** — 层次化多注意力序列建模：

```
输入层:  预训练特征 -> nn.Embedding -> Linear 投影 (异构->256d)
序列层:  3 序列 (Sx/Sy/Sxy) x 3 模态 (ID/Image/Text) = 9 个 DomainTransformer
预测层:  余弦相似度 x 温度缩放 -> 可学习模态加权融合 -> 域掩码 -> 可学习跨域聚合
```

- **DomainTransformer**: 2 层 4 头 Pre-LN Transformer，正弦位置编码，因果+填充双重掩码
- **晚融合**: 各模态独立计算余弦分数，可学习权重加权求和
- **跨域聚合**: 可学习 lambda1/lambda2 (初始 0.3/0.1)
- **损失**: L = L_Sx + lambda1 x L_Sy + lambda2 x L_Sxy

### 消融配置

| 模式 | 模态 | 跨域 | 参数量 |
|------|------|:---:|:---:|
| id_only | ID | - | 27.9M |
| id_text | ID + Text | - | 78.6M |
| id_text_img | ID + Text + Image | - | 117.8M |
| full | ID + Text + Image | Yes | 126.3M |

## 训练配置

| 超参数 | 值 |
|------|------|
| 优化器 | AdamW (lr=1.4e-3, wd=0.01) |
| 调度器 | ReduceLROnPlateau (mode="max", patience=3) |
| 早停 | patience=5 |
| Batch Size | 256-768 (按参数量自适应) |
| AMP | 是，梯度裁剪 1.0 |
| GPU | RTX 5090 32GB (AutoDL) |
| 评测 | 全量排序 (43,528 候选)，HR@10 / NDCG@10 / MRR |

## 主要实验结果

| 实验 | 关键发现 |
|------|------|
| 架构消融 | +Text +0.60% / +Image -0.01% / +Cross -0.35% (Test HR@10) |
| 文本变体消融 | LLM 增强 +0.08%，极差 0.18%，增益源于 BGE 编码器 |
| 推理策略消融 | in_only 6.11% vs full 3.36%，跨域信号为噪声 |
| 最优策略 | full 架构训练 + in_only 推理，Test HR@10=6.11% |

## 环境依赖

```
torch>=2.0
transformers
pandas numpy tqdm Pillow scipy
openai (DeepSeek API)
```

## 参考

基于 LLM-EMF: *LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation*
