import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 正弦位置编码——注入位置信息，区分序列元素顺序  
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[-x.size(1):].unsqueeze(0)

# 单序列 × 单模态的 Transformer Encoder
class DomainTransformer(nn.Module):

    def __init__(self, input_dim, d_model=256, n_heads=4, n_layers=2, dropout=0.2, max_len=200):
        super().__init__()
        # 不同维度的异构模态特征（ID 512 维、图像 768 维、文本 1024 维）统一映射到 d_model=256 维
        self.input_proj = nn.Linear(input_dim, d_model)
        # 位置信息 + Dropout
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, seq_emb, mask):
        x = self.input_proj(seq_emb)
        x = self.pos_enc(x)
        x = self.dropout(x)
        # 因果编码
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        pad_mask = (mask == 0)
        if causal_mask.dtype != torch.bool:
            causal_mask = causal_mask.bool()
        # 填充掩码
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        x = self.ln(x)
        lengths = (mask.sum(dim=1) - 1).long()
        return x[torch.arange(x.shape[0], device=x.device), lengths.clamp(min=0)]


class CDSRModel(nn.Module):
    def __init__(self, n_items, id_feats, img_feats, tex_feats,
                 movie_mask, book_mask,
                 d_model=256, n_heads=4, n_layers=2, dropout=0.2, max_len=200,
                 ablation="full"):
        super().__init__()
        # 消融部分配置
        self.ablation = ablation
        self.id_dim = id_feats.shape[1]
        self.img_dim = img_feats.shape[1]
        self.tex_dim = tex_feats.shape[1]

        ablation_configs = {
            "id_only":      {"modalities": ["id"],              "cross_domain": False},
            "id_text":      {"modalities": ["id", "tex"],       "cross_domain": False},
            "id_text_img":  {"modalities": ["id", "tex", "img"], "cross_domain": False},
            "full":         {"modalities": ["id", "tex", "img"], "cross_domain": True},
        }
        cfg = ablation_configs.get(ablation)
        if cfg is None:
            raise ValueError(f"无 ablation mode: {ablation}. "
                             f"必须有一 {list(ablation_configs.keys())}")
        # 模态 + 跨域配置
        self.enabled_modalities = cfg["modalities"]
        self.cross_domain = cfg["cross_domain"]

        # Embedding 多模态嵌入表（异构维度）
        self.id_emb = nn.Embedding(n_items, self.id_dim)
        self.id_emb.weight.data.copy_(id_feats)
        self.id_proj = nn.Linear(self.id_dim, d_model)
        self.mod_dims = {"id": self.id_dim}

        if "tex" in self.enabled_modalities:
            self.tex_emb = nn.Embedding(n_items, self.tex_dim)
            self.tex_emb.weight.data.copy_(tex_feats)
            self.tex_proj = nn.Linear(self.tex_dim, d_model)
            self.mod_dims["tex"] = self.tex_dim

        if "img" in self.enabled_modalities:
            self.img_emb = nn.Embedding(n_items, self.img_dim)
            self.img_emb.weight.data.copy_(img_feats)
            self.img_proj = nn.Linear(self.img_dim, d_model)
            self.mod_dims["img"] = self.img_dim

        # 域掩码
        self.register_buffer("movie_mask", movie_mask)
        self.register_buffer("book_mask", book_mask)

        # DomainTransformers（各模态用独立 input_dim）
        dt_kwargs = dict(d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, dropout=dropout, max_len=max_len)
        sequences = ["X", "Y"] + (["XY"] if self.cross_domain else [])
        # 批量创建 Transformer
        # 序列 + 模态组合：Sx/Sy/Sx+y × ID/Text/Image
        for seq in sequences:
            for mod in self.enabled_modalities:
                setattr(self, f"tf_{seq}_{mod}",
                        DomainTransformer(self.mod_dims[mod], **dt_kwargs))

        # 融合权重 ： 多模态可学习权重
        n_mod = len(self.enabled_modalities)
        if n_mod >= 2:
            self.logit_a = nn.Parameter(torch.tensor(0.0))
        if n_mod >= 3:
            self.logit_b = nn.Parameter(torch.tensor(0.0))

        # 余弦温度缩放：训练稳定
        self.log_temp = nn.Parameter(torch.tensor(2.3026))

    # 单序列多模态编码器：输入 ID/Image/Text 序列 + 掩码，输出融合后的用户偏好向量
    def _encode_sequence(self, seq_name, ids, mask):
        result = {}
        valid = (mask.sum(dim=1) > 0)
        safe_mask = mask.clone()
        if not valid.all():
            safe_mask[~valid, 0] = 1

        for mod in self.enabled_modalities:
            emb_table = getattr(self, f"{mod}_emb")
            emb = emb_table(ids)
            tf = getattr(self, f"tf_{seq_name}_{mod}")
            h = tf(emb, safe_mask)
            if not valid.all():
                h = torch.where(valid.unsqueeze(-1), h, torch.zeros_like(h))
            result[mod] = h

        return result
    # 多模态融合预测：输入用户偏好向量 + 全量物品嵌入，计算加权余弦相似度得到预测分数
    def _fuse(self, h_dict, domain_mask, all_dict):
        # 余弦相似度
        def cosine(a, b):
            return F.normalize(a, p=2, dim=-1, eps=1e-4) @ F.normalize(b, p=2, dim=-1, eps=1e-4).T

        scale = torch.exp(self.log_temp)
        sims = []
        # 多模态加权融合
        for mod in self.enabled_modalities:
            sim = scale * cosine(h_dict[mod], all_dict[mod])
            sims.append(sim)

        n = len(sims)
        # 单模态直接返回
        if n == 1:
            return sims[0].masked_fill(~domain_mask.unsqueeze(0), float('-inf'))
        # 多模态：用可学习权重经 softmax 归一化后，加权求和各模态的分数
        if n == 2:
            w = F.softmax(torch.stack([
                self.logit_a,
                torch.tensor(0.0, device=self.logit_a.device)
            ]), dim=0)
            result = w[0] * sims[0] + w[1] * sims[1]
            return result.masked_fill(~domain_mask.unsqueeze(0), float('-inf'))

        if n == 3:
            w = F.softmax(torch.stack([
                self.logit_a, self.logit_b,
                torch.tensor(0.0, device=self.logit_a.device)
            ]), dim=0)
            result = w[0] * sims[0] + w[1] * sims[1] + w[2] * sims[2]
            return result.masked_fill(~domain_mask.unsqueeze(0), float('-inf'))
        
    # 前向传播：输入交互序列 + 掩码，输出预测分数。返回跨域预测（Sy→X、Sx→Y、Sxy→X/Y）
    def forward(self, sx, sx_mask, sy, sy_mask, sxy, sxy_mask, return_cross=False):
        all_dict = {}
        for mod in self.enabled_modalities:
            emb = getattr(self, f"{mod}_emb")
            proj = getattr(self, f"{mod}_proj")
            all_dict[mod] = proj(emb.weight)

        hX = self._encode_sequence("X", sx, sx_mask)
        hY = self._encode_sequence("Y", sy, sy_mask)

        full_mask = torch.ones(len(self.movie_mask), dtype=torch.bool, device=sx.device)

        out = {
            'P_X': self._fuse(hX, self.movie_mask, all_dict),
            'P_Y': self._fuse(hY, self.book_mask, all_dict),
        }
        # 跨域：混合编码（Sx+y） + 融合预测
        if self.cross_domain:
            hXY = self._encode_sequence("XY", sxy, sxy_mask)
            out['P_XY'] = self._fuse(hXY, full_mask, all_dict)
            # 跨域迁移预测：Sx+y → X/Y 和 Sx→Y、Sy→X
            if return_cross:
                out.update({
                    'P_Y_to_X': self._fuse(hY, self.movie_mask, all_dict),
                    'P_X_to_Y': self._fuse(hX, self.book_mask, all_dict),
                    'P_XY_to_X': self._fuse(hXY, self.movie_mask, all_dict),
                    'P_XY_to_Y': self._fuse(hXY, self.book_mask, all_dict),
                })
        elif return_cross:
            out.update({
                'P_Y_to_X': self._fuse(hY, self.movie_mask, all_dict),
                'P_X_to_Y': self._fuse(hX, self.book_mask, all_dict),
            })

        return out
"""
CDSR 模型 — 层次化多注意力序列建模

架构：3 交互序列 (Sx/Sy/Sx+y) × 3 模态 (ID/Image/Text) = 9 个 DomainTransformer
     预测层晚融合 + 跨域聚合

预测逻辑：每个编码器输出一个用户偏好向量，和全量物品嵌入计算余弦相似度得到预测分数，
        最后按可学习权重加权融合。

特征维度 (异构):
  - ID:    512d (LightGCN 3-layer)
  - Image: 768d (ViT-L/14)
  - Text:  1024d (BGE-large-en-v1.5, 三变体）

消融模式 (ablation 参数):
  - "id_only"      仅 ID embedding，单域 (Sx+Sy)，2 个 Transformer
  - "id_text"      ID + Text，单域，4 个 Transformer
  - "id_text_img"  ID + Text + Image，单域，6 个 Transformer
  - "full"         ID + Text + Image + 跨域 (Sxy+聚合)，9 个 Transformer（默认）
"""