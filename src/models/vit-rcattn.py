import torch
import torch.nn as nn
from .crisscross_attention import CrissCrossAttention
from .edge_token_module import EdgeTokenExtractor


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = CrissCrossAttention(dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.ff(self.norm(x))
        return x


class BoundaryAwareViT(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depth, num_heads):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, embed_dim))

        self.edge_extractor = EdgeTokenExtractor(embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed

        edge_tokens = self.edge_extractor(x)
        x = x + edge_tokens

        for blk in self.layers:
            x = blk(x)

        x = self.head(x)
        b, p, _ = x.shape
        h = w = int(p ** 0.5)
        return x.transpose(1, 2).reshape(b, 1, h, w)
