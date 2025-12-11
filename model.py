# test/model.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class PatchEmbeddings(nn.Module):
    def __init__(self, num_rois, embed_dim):
        super().__init__()
        self.num_rois = num_rois
        self.embed_dim = embed_dim
        self.fc1_proj = nn.Linear(num_rois, embed_dim)
        self.fc2_proj = nn.Linear(num_rois, embed_dim)
        self.sc1_proj = nn.Linear(num_rois, embed_dim)
        self.sc2_proj = nn.Linear(num_rois, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embeddings = nn.Parameter(torch.zeros(1, num_rois + 1, embed_dim))
        logger.info(f"初始化PatchEmbeddings: 输入维度={num_rois}, 嵌入维度={embed_dim}")

    def forward(self, fc1_patches, fc2_patches, sc1_patches, sc2_patches):
        batch_size = fc1_patches.shape[0]
        actual_size = fc1_patches.shape[1]
        if actual_size != self.num_rois:
            # 如果尺寸不一致，尝试广播或截断/填充（此处简化为截断或重复）
            if actual_size > self.num_rois:
                fc1_patches = fc1_patches[:, :self.num_rois]
                fc2_patches = fc2_patches[:, :self.num_rois]
                sc1_patches = sc1_patches[:, :self.num_rois]
                sc2_patches = sc2_patches[:, :self.num_rois]
            else:
                # pad zeros
                pad = self.num_rois - actual_size
                zero = torch.zeros(batch_size, pad, device=fc1_patches.device, dtype=fc1_patches.dtype)
                fc1_patches = torch.cat([fc1_patches, zero], dim=1)
                fc2_patches = torch.cat([fc2_patches, zero], dim=1)
                sc1_patches = torch.cat([sc1_patches, zero], dim=1)
                sc2_patches = torch.cat([sc2_patches, zero], dim=1)

        fc1_embedded = self.fc1_proj(fc1_patches)
        fc2_embedded = self.fc2_proj(fc2_patches)
        sc1_embedded = self.sc1_proj(sc1_patches)
        sc2_embedded = self.sc2_proj(sc2_patches)

        fc1_cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        fc2_cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sc1_cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sc2_cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        z_fc1 = torch.cat((fc1_cls_tokens, fc1_embedded), dim=1) + self.pos_embeddings
        z_fc2 = torch.cat((fc2_cls_tokens, fc2_embedded), dim=1) + self.pos_embeddings
        z_sc1 = torch.cat((sc1_cls_tokens, sc1_embedded), dim=1) + self.pos_embeddings
        z_sc2 = torch.cat((sc2_cls_tokens, sc2_embedded), dim=1) + self.pos_embeddings

        return z_fc1, z_fc2, z_sc1, z_sc2


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.layer_scale1 = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.1)
        self.layer_scale2 = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.1)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_output, _ = self.attn(x1, x1, x1)
        x = x + self.layer_scale1.transpose(0, 1) * attn_output
        x2 = self.norm2(x)
        mlp_output = self.mlp(x2)
        x = x + self.layer_scale2.transpose(0, 1) * mlp_output
        return x


class SCG_ViT(nn.Module):
    def __init__(self, num_rois, embed_dim, num_heads, num_layers, k, num_classes, dropout=0.1):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(num_rois, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, max(1, embed_dim // 2)),
            nn.GELU(),
            nn.LayerNorm(max(1, embed_dim // 2)),
            nn.Dropout(dropout * 0.75),
            nn.Linear(max(1, embed_dim // 2), max(1, embed_dim // 4)),
            nn.GELU(),
            nn.LayerNorm(max(1, embed_dim // 4)),
            nn.Dropout(dropout * 0.5),
            nn.Linear(max(1, embed_dim // 4), num_classes)
        )
        self.k = k

    def forward(self, fc1_data, fc2_data, sc1_data, sc2_data):
        z_fc1, z_fc2, z_sc1, z_sc2 = self.patch_embeddings(fc1_data, fc2_data, sc1_data, sc2_data)
        # 简化版：拼接四个模态的补丁嵌入，按 transformer 逐层处理
        combined = torch.cat([z_fc1, z_fc2, z_sc1, z_sc2], dim=1)  # (B, 4*(N+1), E)
        x = combined.transpose(0, 1)  # (S, B, E)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        cls_output = x[0]  # (B, E)
        logits = self.head(cls_output)
        return logits