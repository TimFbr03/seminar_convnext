import torch
import torch.nn as nn


class AblationBlock(nn.Module):
    def __init__(self, dim, single_activation=False,
                 inverted_bottleneck=False, layer_norm=False):
        super().__init__()
        self.single_activation = single_activation
        self.inverted_bottleneck = inverted_bottleneck
        self.layer_norm = layer_norm

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Norm: LayerNorm erwartet (B,H,W,C), BatchNorm (B,C,H,W)
        self.norm = nn.LayerNorm(dim) if layer_norm else nn.BatchNorm2d(dim)

        if inverted_bottleneck:
            self.expand  = nn.Linear(dim, 4 * dim)
            self.act     = nn.GELU()
            self.project = nn.Linear(4 * dim, dim)
        else:
            self.pwconv1 = nn.Linear(dim, dim)
            self.pwconv2 = nn.Linear(dim, dim)
            self.act     = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)

        # → channel-last für Norm + MLP
        x = x.permute(0, 2, 3, 1)  # (B,C,H,W) → (B,H,W,C)

        # Norm (LayerNorm oder BN — BN braucht channel-first,
        # also kurz zurück wenn nötig)
        if self.layer_norm:
            x = self.norm(x)
        else:
            x = x.permute(0, 3, 1, 2)   # → (B,C,H,W)
            x = self.norm(x)
            x = x.permute(0, 2, 3, 1)   # → (B,H,W,C)

        # MLP in channel-last
        if self.inverted_bottleneck:
            x = self.expand(x)
            x = self.act(x)
            x = self.project(x)
        else:
            x = self.pwconv1(x)
            if not self.single_activation:
                x = self.act(x)
            x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)  # → (B,C,H,W)
        return x + residual
