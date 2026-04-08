import torch
import torch.nn as nn


class EdgeTokenExtractor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        h = w = int(n ** 0.5)

        x2d = x.view(b, h, w, c)
        gx = x2d[:, :, 1:] - x2d[:, :, :-1]
        gy = x2d[:, 1:, :] - x2d[:, :-1, :]

        gx = torch.nn.functional.pad(gx, (0, 0, 0, 0, 0, 1))
        gy = torch.nn.functional.pad(gy, (0, 0, 0, 1, 0, 0))

        edge = torch.sqrt(gx ** 2 + gy ** 2)
        edge = edge.view(b, n, c)

        return self.edge_proj(edge)
