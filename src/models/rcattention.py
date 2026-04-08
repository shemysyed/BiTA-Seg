import torch
import torch.nn as nn


class RCAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        q = self.q(x).view(B, H, W, C)
        k = self.k(x).view(B, H, W, C)
        v = self.v(x).view(B, H, W, C)

        row = torch.einsum("bhwc, bhOc -> bhwO", q, k.transpose(-1, -2))
        col = torch.einsum("bhwc, bHwc -> bhwH", q, k)

        attn = torch.softmax(row + col, dim=-1)

        out = torch.einsum("bhwo, bhoc -> bhwc", attn, v)
        return out.view(B, N, C)
