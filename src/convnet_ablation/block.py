import torch
import torch.nn as nn


class AblationBlock(nn.Module):

    def __init__(
        self,
        dim,
        single_activation=False,
        inverted_bottleneck=False,
        layer_norm=False
    ):
        super().__init__()

        self.single_activation = single_activation
        self.inverted_bottleneck = inverted_bottleneck
        self.layer_norm = layer_norm

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )

        if layer_norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.BatchNorm2d(dim)

        if inverted_bottleneck:

            self.expand = nn.Linear(dim, 4 * dim)
            self.act = nn.GELU()
            self.project = nn.Linear(4 * dim, dim)

        else:

            self.conv1 = nn.Conv2d(dim, dim, 1)
            self.conv2 = nn.Conv2d(dim, dim, 1)

            self.act1 = nn.GELU()
            self.act2 = nn.GELU()

    def forward(self, x):

        residual = x

        x = self.dwconv(x)

        if self.layer_norm:

            x = x.permute(0, 2, 3, 1)

            x = self.norm(x)

            if self.inverted_bottleneck:

                x = self.expand(x)

                x = self.act(x)

                x = self.project(x)

            x = x.permute(0, 3, 1, 2)

        else:

            x = self.norm(x)

            if self.inverted_bottleneck:

                x = x.permute(0, 2, 3, 1)

                x = self.expand(x)

                x = self.act(x)

                x = self.project(x)

                x = x.permute(0, 3, 1, 2)

            else:

                x = self.conv1(x)

                if not self.single_activation:
                    x = self.act1(x)

                x = self.conv2(x)

                x = self.act2(x)

        return x + residual
