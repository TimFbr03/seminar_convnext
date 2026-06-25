import torch
import torch.nn as nn

from block import AblationBlock

class FashionConvNeXt(nn.Module):

    def __init__(
        self,
        in_channels=3,   # ← neu: STL-10 = 3, Fashion-MNIST = 1
        dim=32,
        num_classes=10,
        patchify=False,
        single_activation=False,
        inverted_bottleneck=False,
        layer_norm=False
    ):
        super().__init__()

        if patchify:

            self.stem = nn.Conv2d(
                in_channels,
                dim,
                kernel_size=4,
                stride=4
            )

        else:

            self.stem = nn.Conv2d(
                in_channels,
                dim,
                kernel_size=3,
                padding=1
            )

        self.blocks = nn.Sequential(
            AblationBlock(
                dim,
                single_activation,
                inverted_bottleneck,
                layer_norm
            ),
            AblationBlock(
                dim,
                single_activation,
                inverted_bottleneck,
                layer_norm
            ),
            AblationBlock(
                dim,
                single_activation,
                inverted_bottleneck,
                layer_norm
            )
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):

        x = self.stem(x)

        x = self.blocks(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return self.head(x)
