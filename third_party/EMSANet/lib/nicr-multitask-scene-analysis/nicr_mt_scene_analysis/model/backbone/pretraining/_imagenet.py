# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
from torch import Tensor
from torch import nn

from ...initialization import he_initialization
from ...initialization import zero_residual_initialization
from ..base import Backbone


class ImageNetClassifier(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        n_classes: int = 1000,
        zero_init_residual: bool = False,
        he_init: bool = True
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(backbone.stages_n_channels[-1], n_classes)

        # default initialization
        if he_init:
            he_initialization(self, debug=True)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual
        # block behaves like an identity. This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            zero_residual_initialization(self, debug=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
