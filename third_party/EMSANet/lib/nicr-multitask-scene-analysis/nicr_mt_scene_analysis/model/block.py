# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Parts of this code are taken and adapted from official torchvision ResNet:
    https://github.com/pytorch/vision/blob/v0.4.2/torchvision/models/resnet.py
    https://github.com/pytorch/vision/blob/v0.11.0/torchvision/models/resnet.py
and ERFNet:
    https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
"""
from typing import Any, Optional, Type, Union

from torch import nn
from torch import Tensor

from .activation import get_activation_class
from .normalization import get_normalization_class
from .utils import conv1x1
from .utils import conv3x3
from ..utils import partial_class


KNOWN_BLOCKS = (
    'basicblock',     # basic block of the ResNet v1
    'bottleneck',     # bottleneck block of ResNet v1.5
    'nonbottleneck1d'    # basic block but with spatially factorized convs
)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        **kwargs
    ) -> None:
        """Basic Block of ResNet v1"""
        super().__init__()

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and "
                             "base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in "
                                      "BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = normalization(planes)
        self.act1 = activation()
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = normalization(planes)
        self.act2 = activation()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3
    # convolution(self.conv2) while original implementation places the stride
    # at the first 1x1 convolution(self.conv1) according to "Deep residual
    # learning for image recognition"https://arxiv.org/abs/1512.03385. This
    # variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        **kwargs
    ) -> None:
        """Bottleneck Block of ResNet v1.5"""
        super().__init__()

        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.norm1 = normalization(width)
        self.act1 = activation()
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.norm2 = normalization(width)
        self.act2 = activation()
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.norm3 = normalization(planes * self.expansion)
        self.act3 = activation()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)

        return out


class NonBottleneck1D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        dropout_p: float = 0.2,
        **kwargs
    ) -> None:
        """NonBottleneck1D block of ERFNet"""
        super().__init__()

        if groups != 1 or base_width != 64:
            raise ValueError("NonBottleneck1D only supports groups=1 and "
                             "base_width=64")

        self.conv1_1 = nn.Conv2d(inplanes, planes, (3, 1),
                                 stride=(stride, 1), padding=(1, 0),
                                 bias=True)
        self.act1_1 = activation()
        self.conv1_2 = nn.Conv2d(planes, planes, (1, 3),
                                 stride=(1, stride), padding=(0, 1),
                                 bias=False)
        self.norm1 = normalization(planes)
        self.act1_2 = activation()

        self.conv2_1 = nn.Conv2d(planes, planes, (3, 1),
                                 padding=(1 * dilation, 0), bias=True,
                                 dilation=(dilation, 1))
        self.act2_1 = activation()
        self.conv2_2 = nn.Conv2d(planes, planes, (1, 3),
                                 padding=(0, 1*dilation), bias=False,
                                 dilation=(1, dilation))
        self.norm2 = normalization(planes)
        self.act2_2 = activation()

        # note, this is channel-wise and no spatial Dropout
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.downsample = downsample
        self.stride = stride
        self.dropout_p = dropout_p

    def forward(self, input):
        output = self.conv1_1(input)
        output = self.act1_1(output)
        output = self.conv1_2(output)
        output = self.norm1(output)
        output = self.act1_2(output)

        output = self.conv2_1(output)
        output = self.act2_1(output)
        output = self.conv2_2(output)
        output = self.norm2(output)

        if self.dropout_p > 0:
            output = self.dropout(output)

        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        return self.act2_2(output + identity)


BlockType = Union[BasicBlock, Bottleneck, NonBottleneck1D]


def get_block_class(
    name: Optional[str] = None,
    **kwargs: Any
) -> Type[BlockType]:
    # global default
    if name is None:
        name = 'nonbottleneck1d'

    name = name.lower()
    if name not in KNOWN_BLOCKS:
        raise ValueError(f"Unknown block: '{name}'")

    if 'basicblock' == name:
        block = BasicBlock
    elif 'bottleneck' == name:
        block = Bottleneck
    elif 'nonbottleneck1d' == name:
        block = NonBottleneck1D

    return partial_class(block, **kwargs)
