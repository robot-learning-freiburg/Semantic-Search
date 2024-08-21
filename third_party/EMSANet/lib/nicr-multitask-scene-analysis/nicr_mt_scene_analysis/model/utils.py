# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Type, Union

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


from .activation import get_activation_class
from .normalization import get_normalization_class


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        kernel_size: int = 1,
        dilation: int = 1,
        stride: int = 1,
        normalization: Union[Type[nn.Module], None] = get_normalization_class(),
        activation: Union[Type[nn.Module], None] = get_activation_class()
    ) -> None:
        super().__init__()

        # create modules
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(n_channels_in, n_channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=normalization is None,
                                          dilation=dilation,
                                          stride=stride))
        if normalization is not None:
            self.add_module('norm', normalization(n_channels_out))

        if activation is not None:
            self.add_module('act', activation())


class SqueezeAndExcitation(nn.Module):
    def __init__(
        self,
        n_channels: int,
        reduction: int = 16,
        activation: Type[nn.Module] = get_activation_class()
    ) -> None:
        super().__init__()

        n_channels_red = n_channels // reduction
        assert n_channels_red > 0

        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, n_channels_red, kernel_size=1),
            activation(),
            nn.Conv2d(n_channels_red, n_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.layers(weighting)
        y = x * weighting
        return y


class SqueezeAndExcitationTensorRT(SqueezeAndExcitation):
    def forward(self, x: Tensor):
        # TensorRT restricts the maximum kernel size for pooling operations
        # by "MAX_KERNEL_DIMS_PRODUCT", which leads to problems if the input
        # feature maps are of large spatial size
        # -> workaround: use cascaded two-staged pooling
        # see: https://github.com/onnx/onnx-tensorrt/issues/333
        if x.shape[2] > 120 and x.shape[3] > 160:
            weighting = F.adaptive_avg_pool2d(x, 4)
        else:
            weighting = x
        weighting = F.adaptive_avg_pool2d(weighting, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
