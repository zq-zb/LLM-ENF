# LLM-EMF · 多模态跨域序列推荐系统

社会计算课程课题 —— 基于 **LLM-EMF** 框架的电影—图书跨域序列推荐系统，融合 LLM 文本语义增强 + BGE/ViT 多模态编码 + Transformer 分层注意力序列建模。

---

## 项目简介

本系统通过用户在**一个领域**（如电影）的历史交互行为，利用多模态深度学习模型，为他在**另一个领域**（如书籍）推荐可能感兴趣的内容，实现**跨域知识迁移与偏好预测**。

核心思路：将用户行为序列投影到统一的多模态表征空间，使用 9 个 DomainTransformer 编码器捕捉时序模式，通过可学习的模态融合策略（ID + 文本 + 图像）评估候选物品的匹配得分。

---

## 数据集

Amazon Reviews 2023，电影 (Movies_and_TV) ↔ 图书 (Books)。

| 指标 | 数值 |
|------|------:|
| 用户数 | 20,030 |
| 物品数 | 43,528（电影 21,280 + 图书 22,248） |
| 总交互 | 688,010 |
| 划分 | Train 647,950 / Val 20,030 / Test 20,030 (leave-last-2) |

---

## 技术架构

```
                    Flask Web 层
   ┌──────────┐  ┌──────────┐  ┌─────────────────┐
   │ 搜索服务  │  │ 推荐服务  │  │ 用户画像 / 社交  │
   └────┬─────┘  └────┬─────┘  └───────┬─────────┘
        └──────────────┼───────────────┘
                       │
               RecEngine (推理引擎)
   ┌───────────────────────────────────────────────┐
   │           CDSRModel (PyTorch)                  │
   │  ┌──────────┐ ┌──────────┐ ┌──────────┐      │
   │  │ ID Emb   │ │ Text Emb │ │ Img Emb  │      │
   │  │ LightGCN │ │ BGE-large │ │ ViT-L/14 │      │
   │  │ 512d     │ │ 1024d    │ │ 768d     │      │
   │  └────┬─────┘ └────┬─────┘ └────┬─────┘      │
   │       └─────────────┼────────────┘             │
   │                     │                          │
   │    DomainTransformer ×9 (d=256, 4head, 2层)    │
   │                     │                          │
   │   余弦相似度 → 可学习模态融合 → 跨域得分聚合     │
   └───────────────────────────────────────────────┘
```

### 模型细节

| 组件 | 规格 | 说明 |
|------|------|------|
| ID Embedding | 43,528 × 512 | LightGCN 3-layer 图协同过滤嵌入 (BPR) |
| Text Embedding | 43,528 × 1024 | BGE-large-en-v1.5 多语言语义嵌入 |
| Image Embedding | 43,528 × 768 | ViT-L/14 视觉编码器 (24层, 14×14 patch) |
| DomainTransformer | d=256, nhead=4, 2层 Pre-LN | 9 个独立编码器（X/Y/XY × ID/Txt/Img） |
| 融合策略 | 可学习 Sigmoid 门控 | 各模态独立计算余弦分数，加权求和 |
| 跨域聚合 | 可学习 λ₁/λ₂ (初始 0.3/0.1) | L = L_Sx + λ₁·L_Sy + λ₂·L_Sxy |
| 温度参数 | 可学习 log_temp | 调节 Softmax 得分锐度 |

### 消融配置

| 模式 | 模态 | 跨域 | 参数量 |
|------|------|:---:|:---:|
| `id_only` | ID | — | 27.9M |
| `id_text` | ID + Text | — | 78.6M |
| `id_text_img` | ID + Text + Image | — | 117.8M |
| `full` | ID + Text + Image | ✓ | 126.3M |

---

## 项目结构

```
LLM-ENF/
├── demo/                              # Web 演示应用
│   ├── app.py                         # Flask 后端 (API 路由 + 服务启动)
│   ├── engine.py                      # 推荐引擎 (模型加载 + 推理 + 相似度)
│   └── templates/
│       ├── base.html                  # Jinja2 母版 (导航栏 + 样式基础)
│       └── index.html                 # 主界面 (跨域推荐 + 社交发现)
│
├── final/                             # 最终特征与数据
│   ├── features/                      # 预计算特征 (npy + csv)
│   │   ├── item_id_lightgcn_512.npy   # LightGCN ID 嵌入
│   │   ├── text_bge_desc_1024.npy     # BGE 描述嵌入 (变体A)
│   │   ├── text_bge_llm_1024.npy      # BGE LLM 增强嵌入 (变体B)
│   │   ├── text_bge_llmdesc_1024.npy  # BGE LLM+描述嵌入 (变体C)
│   │   ├── image_features_768.npy     # ViT-L/14 视觉特征
│   │   ├── image_id_map_vitl.csv      # 图片ID映射
│   │   └── text_id_map_bge_desc.csv   # 文本ID映射
│   ├── images/                        # 物品图片 (43,509 张, ~3.3 GB)
│   ├── train.csv / val.csv / test.csv # 训练/验证/测试集
│   ├── item_meta_merged.csv           # 合并元数据
│   ├── item_meta_llm_enhanced_v2.csv  # LLM 增强元数据
│   ├── item2id.json / user2id.json    # ID 映射表
│   └── item_image_map.json            # 物品-图片映射
│
├── LLM+特征提取/                       # LLM 增强 + 多模态特征提取
│   ├── vitl14提取/                     # 当前特征管线
│   │   ├── text_llm_enhance_v2.ipynb       # LLM V2 六维度语义增强 (DeepSeek)
│   │   ├── extract_id_lightgcn.ipynb       # LightGCN 512d ID 协同特征
│   │   ├── extract_image_vitl14.ipynb      # ViT-L/14 768d 图像特征
│   │   └── extract_text_bge_3var.ipynb     # BGE 1024d 三变体文本特征
│   └── CLIP特征提取/                    # 旧版 CLIP 管线 (已弃用)
│       ├── text_llm_enhance.ipynb
│       ├── extract_id_embeddings.ipynb
│       ├── extract_image_features.ipynb
│       └── extract_text_features.ipynb
│
├── 训练/                               # 模型训练与消融实验
│   ├── 主实验训练/                      # 主实验
│   │   ├── cdsr_model.py              # 模型定义
│   │   ├── train.ipynb                # 训练脚本
│   │   ├── ablation.ipynb             # 架构消融 (4 变体)
│   │   ├── ablation_text_variants.ipynb # 文本变体消融 (3 变体)
│   │   └── requirements.txt
│   ├── 主实验+图文对齐/                 # +InfoNCE 图文对齐 Loss
│   │   ├── cdsr_model.py
│   │   └── train.ipynb
│   └── CLIP特征训练/                    # 旧版 CLIP 特征训练 (已弃用)
│
├── pre处理/                            # 数据预处理
│   ├── pre_meta_movie.ipynb           # 电影元数据清洗
│   ├── pre_meta_book.ipynb            # 图书元数据清洗
│   ├── pre_interaction_movie.ipynb    # 电影交互过滤
│   ├── pre_interaction_book.ipynb     # 图书交互过滤
│   ├── merge_and_split.ipynb          # 双域合并 + 交叉筛选 + 时序划分
│   └── download_images.py             # 批量下载物品图片
│
├── movie_book_cdsr_processed/          # 预处理后数据 (中间产物)
│
├── old版本/                            # 旧版实验代码
│
├── best_model.pt                       # 最优模型权重 (~484 MB, Git LFS)
├── best_val_model.pt                   # 最优验证模型权重 (~484 MB, Git LFS)
├── inference.py                        # 推理脚本
└── README.md                           # 本文件
```

---

## 特征工程

### 多模态特征体系

| 模态 | 方法 | 维度 | 说明 |
|------|------|:---:|------|
| ID 协同 | LightGCN 3-layer | 512 | BPR 损失，weight_decay=0 |
| 图像 | ViT-L/14 | 768 | 24 层 Transformer，14×14 patch |
| 文本 | BGE-large-en-v1.5 | 1024 | 512 token，含 LLM 六维度增强 |

### LLM 文本增强

DeepSeek API，六维度结构化 Prompt (Genre / Plot / Style / Themes / Audience / Similar To)，200 词输出，20 线程并发，全量 43,528 物品。

### 文本三变体

| 变体 | 输入 | 用途 |
|------|------|------|
| A: `desc` | title + 原始描述[:1500] | 纯原文基线 |
| B: `llm` | title + LLM 生成描述 | LLM 蒸馏价值 |
| C: `llmdesc` | title + LLM + desc[:400] | LLM+原文互补 |

---

## 快速开始

### 环境依赖

```bash
# Python 3.10+
# GPU: RTX 5090 32GB (推荐), CPU 可推理
pip install torch>=2.0 flask pandas numpy tqdm Pillow transformers
```

### 启动 Web 演示

```bash
cd demo
python app.py
```

服务启动后将打印：
```
初始化推荐引擎...
加载数据...
加载模型...
构建用户索引...
预计算活跃用户嵌入...
初始化完成
启动服务: http://127.0.0.1:5000
```

浏览器访问 `http://127.0.0.1:5000` 即可看到交互界面。

### 命令行推理

```bash
python inference.py
```

---

## API 接口

| 端点 | 方法 | 参数 | 说明 |
|------|------|------|------|
| `/api/search` | GET | `q` (关键词), `domain` (movie/book/all) | 搜索物品 |
| `/api/popular/<domain>` | GET | `limit` (数量) | 热门物品列表 |
| `/api/recommend` | POST | `items` (物品ID数组), `target` (目标域), `top_k` | **核心推荐接口** |
| `/api/users` | GET | — | TOP50 活跃用户画像 |
| `/api/user/<id>` | GET | — | 单个用户详情 |
| `/api/similar_users` | POST | `user_id` 或 `items`, `top_k` | 查找相似用户 |

### 核心推荐流程

```
1. 用户在前端选择 N 个来源域物品 (如看过 5 部电影)
2. POST /api/recommend 携带物品 ID + 目标域
3. 后端 Transformer 编码用户序列 → 融合表征向量
4. 在目标域候选集上批量计算匹配得分
5. 返回 TOP-K 排序结果 (含得分 + 物品元数据)
```

---

## 训练配置

| 超参数 | 值 |
|------|------|
| 优化器 | AdamW (lr=1.4e-3, wd=0.01) |
| 调度器 | ReduceLROnPlateau (mode="max", patience=3) |
| 早停 | patience=5 |
| Batch Size | 256-768 (按参数量自适应) |
| AMP | 是，梯度裁剪 1.0 |
| 评测 | 全量排序 (43,528 候选)，HR@10 / NDCG@10 / MRR |

---

## 关键设计问答

**为什么用 Transformer？**
用户的历史行为本质是一个变长序列，存在时序依赖。RNN 易遗忘早期交互，Transformer 的 Self-Attention 可以更好地建模序列内任意位置物品之间的关联。

**为什么用 9 个 Transformer？**
三个模态（ID / Text / Image）× 三个分组（Sx=源域 / Sy=目标域 / Sxy=混合），各自独立编码后通过平均池化得到统一用户表征。这种设计让模型可以区分不同模态的贡献、分别建模跨域和域内的行为模式，并通过 Sigmoid 门控实现软性模态融合。

**为什么用预训练嵌入？**
LightGCN（图结构）、BGE-large（文本语义）、ViT-L/14（视觉风格）、LLM 增强（外部知识）四种表征互补，覆盖推荐系统中结构、语义、视觉三个核心维度。

---

## 注意事项

- 模型权重文件 ~970 MB（含 9 个 Transformer + 3 个 Embedding 矩阵），使用 Git LFS 管理
- 特征 npy 文件 ~724 MB，图片 ~3.3 GB（Git LFS）
- Web 演示为单进程 Flask 服务，仅供本地演示使用
- 用户搜索为精确子串匹配（非模糊搜索），建议使用物品名称中的关键词
- `movie_book_cdsr_processed/` 和 `old版本/` 为中间产物和历史版本，非核心依赖

---

## 参考文献

| 序号 | 论文 | 用途 |
|:---:|------|------|
| [1] | He X, Deng K, Wang X, et al. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.** *SIGIR 2020.* | ID 协同特征提取 |
| [2] | Radford A, Kim J W, Hallacy C, et al. **Learning Transferable Visual Models From Natural Language Supervision.** *ICML 2021.* | ViT-L/14 图像特征提取 |
| [3] | Xiao S, Liu Z, Zhang P, et al. **C-Pack: Packaged Resources To Advance General Chinese Embedding.** *arXiv:2309.07597, 2023.* | BGE-large 文本特征编码 |
| [4] | Vaswani A, et al. **Attention Is All You Need.** *NeurIPS 2017.* | Transformer 序列编码器 |
| [5] | — **LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation.** | LLM-EMF 主框架 |

> 注：本项目基于 LLM-EMF 框架实现，使用 DeepSeek V4 作为 LLM 文本增强后端，RTX 5090 32GB 训练。

---

*跨域多模态推荐系统 · Powered by Transformer & Graph Embeddings · 社会计算课题演示组*
