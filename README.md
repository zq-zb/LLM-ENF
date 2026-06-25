# CrossDomain RecSys · 跨域多模态推荐系统

社会计算课程课题 —— 基于 Transformer 和多种嵌入表征的跨域（电影 ↔ 书籍）个性化推荐系统。

---

## 项目简介

本系统通过用户在**一个领域**（如电影）的历史交互行为，利用多模态深度学习模型，为他在**另一个领域**（如书籍）推荐可能感兴趣的内容，实现**跨域知识迁移与偏好预测**。

核心思路：将用户行为序列投影到统一的多模态表征空间，使用 Transformer 编码器捕捉时序模式，通过可学习的模态融合策略（ID + 文本 + 图像）评估候选物品的匹配得分。

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
   │         CrossDomainRecModel (PyTorch)          │
   │  ┌──────────┐ ┌──────────┐ ┌──────────┐      │
   │  │ ID Emb   │ │ Text Emb │ │ Img Emb  │      │
   │  │ LightGCN │ │ BGE-M3   │ │ ViT-L    │      │
   │  │ 512d     │ │ 1024d    │ │ 768d     │      │
   │  └────┬─────┘ └────┬─────┘ └────┬─────┘      │
   │       └─────────────┼────────────┘             │
   │                     │                          │
   │    Transformer Encoder ×9 (d=256, 8head, 2层)  │
   │                     │                          │
   │       可学习融合: a·ID + b·Tex + c·Img          │
   └───────────────────────────────────────────────┘
```

### 模型细节

| 组件 | 规格 | 说明 |
|------|------|------|
| ID Embedding | 43,528 × 512 | LightGCN 图协同过滤嵌入 |
| Text Embedding | 43,528 × 1024 | BGE-M3 多语言语义嵌入 |
| Image Embedding | 43,528 × 768 | ViT-L/14 视觉编码器 |
| Transformer | d_model=256, nhead=8, 2层 | 9 个独立编码器（X/Y/XY × ID/Tex/Img）|
| 融合策略 | Sigmoid 门控 | 可学习的 a·ID + b·Tex + (1-a-b)·Img |
| 温度参数 | 可学习 log_temp | 调节 Softmax 得分的锐度 |

### 预训练嵌入语义

- **结构嵌入**：LightGCN 在用户-物品二分图上训练，捕获协同过滤信号
- **语义嵌入**：BGE-M3 对物品标题/描述编码，支持中英文跨语言检索
- **视觉嵌入**：ViT-L 对物品封面/海报编码，捕获视觉风格特征
- **LLM 增强**：大语言模型对物品描述进行内容增强，融合外部知识

---

## 项目结构

```
社会计算/
├── demo/                          # Web 演示应用
│   ├── app.py                     # Flask 后端 (API 路由 + 服务启动)
│   ├── engine.py                  # 推荐引擎 (模型加载 + 推理 + 相似度计算)
│   └── templates/
│       ├── base.html              # Jinja2 母版 (导航栏 + 样式基础)
│       ├── index.html             # 主界面 (跨域推荐 + 社交发现)
│       └── recommend.html         # 备选推荐页 (用户选择 + 过滤器)
│
├── models/                        # 训练好的模型权重
│   ├── best_model.pt              # 最优模型
│   └── best_val_model.pt          # 最优验证模型
│
├── features/                      # 预计算特征文件
│   ├── item_id_lightgcn_512.npy   # LightGCN ID 嵌入
│   ├── text_bge_desc_1024.npy     # BGE 描述嵌入
│   ├── text_bge_llm_1024.npy      # BGE LLM 增强嵌入
│   ├── text_bge_llmdesc_1024.npy  # BGE LLM 描述嵌入
│   ├── image_features_768.npy     # ViT 视觉特征
│   ├── image_id_map_vitl.csv      # 图片ID映射
│   └── text_id_map_bge_desc.csv   # 文本ID映射
│
├── images/                        # 物品图片 (~10,000+ 张)
│
├── train.csv                      # 训练集
├── val.csv                        # 验证集
├── test.csv                       # 测试集
│
├── item_meta_movie.csv            # 电影元数据
├── item_meta_book.csv             # 书籍元数据
├── item_meta_merged.csv           # 合并元数据
├── item_meta_llm_enhanced.csv     # LLM 增强元数据 v1
├── item_meta_llm_enhanced_v2.csv  # LLM 增强元数据 v2
│
├── item2id.json                   # 物品名→ID 映射
├── user2id.json                   # 用户→ID 映射
├── domain2id.json                 # 领域→ID 映射
└── README.md                      # 本文件
```

---

## 快速开始

### 环境依赖

```bash
# Python 3.10+
pip install torch flask pandas numpy
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

## 前端界面

### 双模式设计

**跨域推荐模式**（默认）：
- 电影→书籍 或 书籍→电影 双向跨域切换
- 支持关键词搜索 (300ms 防抖) + 热门浏览
- 多选队列 (上限 50)，以彩色标签直观展示
- Transformer 得分可视化 (进度条 + 百分比 + 排名徽章)

**社交发现模式**：
- TOP50 活跃用户画像列表 (支持 ID 过滤)
- 基于余弦相似度的用户向量匹配
- 共同喜好作品展示 + 品味重叠度量化

### 视觉风格

- 暗黑赛博朋克主题 (紫色/蓝色/绿色渐变)
- 玻璃拟态 (Glassmorphism) 面板设计
- Canvas 动态粒子网络背景 (鼠标交互)
- Toast 通知系统 + 骨架屏加载 + 卡片入场动画

---

## 关键设计问答

**为什么用 Transformer？**
用户的历史行为本质是一个变长序列，存在时序依赖。RNN 易遗忘早期交互，Transformer 的 Self-Attention 可以更好地建模序列内任意位置物品之间的关联。

**为什么用 9 个 Transformer？**
三个模态（ID / Text / Image）× 三个分组（X=源域 / Y=目标域 / XY=混合），各自独立编码后通过平均池化得到统一用户表征。这种设计让模型可以区分不同模态的贡献、分别建模跨域和域内的行为模式，并通过 Sigmoid 门控实现软性模态融合。

**为什么用预训练嵌入？**
LightGCN（图结构）、BGE-M3（文本语义）、ViT-L（视觉风格）、LLM 增强（外部知识）四种表征互补，覆盖推荐系统中结构、语义、视觉三个核心维度。

---

## 注意事项

- 模型权重文件约 900 MB（包含 9 个 Transformer 编码器 + 3 个 Embedding 矩阵），首次加载需等待数秒
- 特征 npy 文件总计约 2 GB，依赖外部预训练模型（BGE-M3, ViT-L）生成
- Web 演示为单进程 Flask 服务，仅供本地演示使用
- 用户搜索为精确子串匹配（非模糊搜索），建议使用物品名称中的关键词

---

*跨域多模态推荐系统 · Powered by Transformer & Graph Embeddings · 社会计算课题演示组*
