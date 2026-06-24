"""
Minimal Vision Transformer for 64×64 grayscale images.

Architecture (chosen to ~match CNN param count of ~503k):
  patch_size=8  →  64 tokens + 1 CLS token = 65 sequence length
  dim=128, depth=4, heads=4, mlp_dim=256
  Total params: ~547k
"""

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # pre-norm + residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer for binary classification on 64×64 grayscale images.

    Args:
        img_size:   Input image height/width (square assumed).
        patch_size: Patch height/width (square assumed).
        in_channels: Number of input channels (1 for grayscale).
        num_classes: Number of output classes.
        dim:        Token embedding dimension.
        depth:      Number of Transformer blocks.
        heads:      Number of attention heads.
        mlp_dim:    Hidden dimension of the MLP inside each block.
        dropout:    Dropout probability.
    """
    def __init__(
        self,
        img_size:    int = 128,
        patch_size:  int = 16,
        in_channels: int = 3,
        num_classes: int = 2,
        dim:         int = 128,
        depth:       int = 4,
        heads:       int = 4,
        mlp_dim:     int = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding: flatten each patch and project to dim
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.patch_size  = patch_size

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        nH, nW = H // p, W // p

        # Robustes Patchify ohne view-nach-unfold
        x = x.reshape(B, C, nH, p, nW, p)       # (B, C, nH, p, nW, p)
        x = x.permute(0, 2, 4, 1, 3, 5)         # (B, nH, nW, C, p, p)
        x = x.reshape(B, nH * nW, C * p * p)    # (B, num_patches, patch_dim)

        x = self.patch_embed(x)                  # (B, N, dim)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)         # (B, N+1, dim)
        x   = self.dropout(x + self.pos_embed)

        x = self.transformer(x)
        x = self.norm(x)

        return self.head(x[:, 0])
