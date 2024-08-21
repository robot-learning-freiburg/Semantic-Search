# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>

Part of this code are taken and adapted from official torchvision ResNet:
    https://github.com/pytorch/vision/blob/v0.4.2/torchvision/models/resnet.py
"""
from typing import Any, List, Optional, Type, Union
import warnings

import torch
from torch import nn
from torch.hub import load_state_dict_from_url

from .base import Backbone
from ..activation import get_activation_class
from ..block import BlockType
from ..block import Bottleneck
from ..normalization import get_normalization_class
from ..utils import conv1x1
from ..utils import SqueezeAndExcitation


MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class ResNetBackbone(Backbone):
    def __init__(
        self,
        block: Type[BlockType],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        n_input_channels: int = 3
    ) -> None:
        super().__init__()

        self._block = block
        self._normalization = normalization
        self._activation = activation

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, "
                             "got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(n_input_channels, self.inplanes,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = normalization(self.inplanes)
        self.act = self._activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # define stages and meta information
        self._stages = [
            nn.Sequential(self.conv1, self.norm1, self.act),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4
        ]
        self._stages_n_channels = [
            64,
            64*self._block.expansion,
            64*2*self._block.expansion,
            64*4*self._block.expansion,
            64*8*self._block.expansion
        ]
        self._stages_downsampling = [
            2,
            4,
            4*2**(1-sum(replace_stride_with_dilation[:1])),
            4*2**(2-sum(replace_stride_with_dilation[:2])),
            4*2**(3-sum(replace_stride_with_dilation))
        ]
        self._stages_memory_layout = [
            'nchw',
            'nchw',
            'nchw',
            'nchw',
            'nchw'
        ]

    @property
    def stages(self) -> List[Union[nn.Sequential, nn.Module]]:
        return self._stages

    @property
    def stages_n_channels(self) -> List[int]:
        return self._stages_n_channels

    @property
    def stages_downsampling(self) -> List[int]:
        return self._stages_downsampling

    @property
    def stages_memory_layout(self) -> List[str]:
        return self._stages_memory_layout

    def _make_layer(
        self,
        block: Type[BlockType],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        normalization = self._normalization
        activation = self._activation
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                normalization(planes*block.expansion),
            )

        layers = []
        layers.append(
            block(inplanes=self.inplanes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  groups=self.groups,
                  base_width=self.base_width,
                  dilation=previous_dilation,
                  normalization=normalization,
                  activation=activation)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(inplanes=self.inplanes,
                      planes=planes,
                      stride=1,
                      downsample=None,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      normalization=normalization,
                      activation=activation)
            )
        return nn.Sequential(*layers)


class ResNetSEBackbone(ResNetBackbone):
    def __init__(
        self,
        block: Type[BlockType],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        n_input_channels: int = 3
    ) -> None:
        super().__init__(
            block=block,
            layers=layers,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            activation=activation,
            n_input_channels=n_input_channels
        )

        # create additional se layers
        # note that the se layers are created after initializing the weights
        # in the base class and thus initialized using commonly used heuristic
        self.se_conv1 = SqueezeAndExcitation(
            n_channels=self.stages_n_channels[0],
            activation=activation
        )
        self.se_layer1 = SqueezeAndExcitation(
            n_channels=self.stages_n_channels[1],
            activation=activation
        )
        self.se_layer2 = SqueezeAndExcitation(
            n_channels=self.stages_n_channels[2],
            activation=activation
        )
        self.se_layer3 = SqueezeAndExcitation(
            n_channels=self.stages_n_channels[3],
            activation=activation
        )
        self.se_layer4 = SqueezeAndExcitation(
            n_channels=self.stages_n_channels[4],
            activation=activation
        )

        # (re)define stages
        self._stages = [
            nn.Sequential(self.conv1, self.norm1, self.act, self.se_conv1),
            nn.Sequential(self.maxpool, self.layer1, self.se_layer1),
            nn.Sequential(self.layer2, self.se_layer2),
            nn.Sequential(self.layer3, self.se_layer3),
            nn.Sequential(self.layer4, self.se_layer4)
        ]


def get_resnet_backbone(
    name: str,
    block: Type[BlockType],
    pretrained_torchvision: bool,
    normalization: Type[nn.Module] = get_normalization_class(),
    activation: Type[nn.Module] = get_activation_class(),
    **kwargs: Any
) -> Union[ResNetBackbone, ResNetSEBackbone]:
    name = name.lower()
    if 'resnet18' in name:
        layers = [2, 2, 2, 2]
    elif 'resnet34' in name or 'resnet50' in name:
        layers = [3, 4, 6, 3]
        if 'resnet50' == name and not issubclass(block, Bottleneck):
            # ResNet50 with BasicBlock or NonBottleneck1D results in ResNet34
            warnings.warn("ResNet50 requires 'Bottleneck' block, but "
                          f"'{block}' was passed")
    elif 'resnet101' in name:
        warnings.warn("ResNet101 requires 'Bottleneck' block, but "
                      f"'{block}' was passed")
        layers = [3, 4, 23, 3]

    Model = ResNetSEBackbone if name.endswith('se') else ResNetBackbone
    model = Model(
        block=block,
        layers=layers,
        normalization=normalization,
        activation=activation,
        **kwargs
    )

    if pretrained_torchvision:
        if name.endswith('se'):
            warnings.warn(f"There are no pretrained weights for {name}. "
                          "Using random weights.")
        else:
            # use weight of torchvision
            state_dict = load_state_dict_from_url(MODEL_URLS[name],
                                                  progress=True)
            # 'bn*' was changed to 'norm*' -> rename keys in state dict
            for key in list(state_dict.keys()):
                new_key = key.replace('bn1', 'norm1')
                new_key = new_key.replace('bn2', 'norm2')
                new_key = new_key.replace('bn3', 'norm3')
                state_dict[new_key] = state_dict.pop(key)

            # final fully-connected layer is removed -> remove keys
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')

            if kwargs.get('n_input_channels', None) == 1:
                # sum the weights of the first convolution
                state_dict['conv1.weight'] = torch.sum(
                    state_dict['conv1.weight'],
                    axis=1,
                    keepdim=True
                )

            print(f"Loading pretrained weights from torchvision for {name}")
            model.load_state_dict(state_dict, strict=True)

    return model
