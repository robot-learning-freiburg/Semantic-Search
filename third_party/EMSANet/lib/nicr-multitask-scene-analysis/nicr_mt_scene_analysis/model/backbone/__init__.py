# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from collections import OrderedDict
import os
from packaging import version
from typing import Any, Optional, Type, Union

import torch
import torchvision
import warnings

from ..activation import get_activation_class
from ..block import BlockType
from ..block import BasicBlock
from ..block import Bottleneck
from ..block import get_block_class
from ..normalization import get_normalization_class
from .base import Backbone
from .resnet import get_resnet_backbone

# Check the torchvision version, as the Swin Transformer V2 models are only
# available in torchvision 0.14.0 and later
IS_SWIN_AVAILABLE = False
if version.parse(torchvision.__version__) >= version.parse('0.14.0'):
    from .swin import get_swin_backbone
    from .swin_multimodal import get_swin_multimodal_backbone
    IS_SWIN_AVAILABLE = True
else:
    warnings.warn("Could not import Swin Transformer V2 models as the "
                  "installed torchvision version is too old. Please update "
                  "torchvision to version 0.14.0 or later.")


KNOWN_BACKBONES = [
    # ResNet (v1)
    'resnet18', 'resnet34', 'resnet50', 'resnet101',
    # ResNet (v1) with dilation in last stage -> downsampling: 16 instead of 32
    'resnet18-d16', 'resnet34-d16', 'resnet50-d16', 'resnet101-d16',
    # ResNet (v1) +SE at the end of each stage
    'resnet18se', 'resnet34se', 'resnet50se', 'resnet101se',
]

if IS_SWIN_AVAILABLE:
    KNOWN_BACKBONES += [
        # Swin Transformer (v1)
        'swin-t', 'swin-s', 'swin-b',
        # Swin Transformer (v2)
        'swin-t-v2', 'swin-s-v2', 'swin-b-v2',
        # Swin Transformer modifications on standard model
        'swin-t-128', 'swin-t-v2-128',
        # Multimodal swin transformer
        'swin-multi-t', 'swin-multi-s', 'swin-multi-b',
        'swin-multi-t-v2', 'swin-multi-s-v2', 'swin-multi-b-v2',
        'swin-multi-t-128', 'swin-multi-t-v2-128'
    ]


def get_backbone(
    name: str,
    resnet_block: Union[str, Type[BlockType]],
    n_input_channels: int = 3,
    normalization: Union[str, Type[torch.nn.Module]] = 'batchnorm',
    activation: Union[str, Type[torch.nn.Module]] = 'relu',
    pretrained: bool = True,
    pretrained_filepath: Optional[str] = None,
    **kwargs: Any
) -> Backbone:
    name = name.lower()

    # get normalization and activation classes
    if isinstance(normalization, str):
        normalization = get_normalization_class(normalization)
    if isinstance(activation, str):
        activation = get_activation_class(activation)

    pretrained_torchvision = pretrained

    if 'resnet' in name:
        # get block class from str
        if isinstance(resnet_block, str):
            resnet_block = get_block_class(resnet_block)

        pretrained_torchvision &= not name.endswith('se')
        pretrained_torchvision &= issubclass(resnet_block, (BasicBlock,
                                                            Bottleneck))

        if 'd16' in name:
            # downsampling: 16 (dilation in last stage)
            replace_stride_with_dilation = [False, False, True]
        else:
            # downsampling: 32
            replace_stride_with_dilation = None

        backbone = get_resnet_backbone(
            name, resnet_block,
            pretrained_torchvision=pretrained_torchvision,
            normalization=normalization,
            activation=activation,
            n_input_channels=n_input_channels,
            replace_stride_with_dilation=replace_stride_with_dilation,
            **kwargs
        )
    elif 'swin' in name:
        if normalization == torch.nn.BatchNorm2d:
            raise ValueError(
                "Swin Transformer does not support BatchNorm2d. "
                "Use LayerNorm instead."
            )
        pretrained_torchvision &= normalization == get_normalization_class('ln')
        pretrained_torchvision &= 'multi' not in name
        pretrained_torchvision &= '-128' not in name
        if 'multi' in name:
            backbone = get_swin_multimodal_backbone(
                name,
                pretrained_torchvision=pretrained_torchvision,
                normalization=normalization,
                n_input_channels=n_input_channels,
                **kwargs
            )
        else:
            backbone = get_swin_backbone(
                name,
                pretrained_torchvision=pretrained_torchvision,
                normalization=normalization,
                n_input_channels=n_input_channels,
                **kwargs
            )
    else:
        raise ValueError(f"Unknown backbone: {name}")

    if pretrained and pretrained_filepath is not None:
        print(f"Loading pretrained weights from: '{pretrained_filepath}'")
        checkpoint = torch.load(pretrained_filepath, map_location='cpu')

        state_dict = OrderedDict()

        # Support for old and new pretrained models
        checkpoint_weights = None
        if 'state_dict' in checkpoint:
            checkpoint_weights = checkpoint['state_dict']
        else:
            checkpoint_weights = checkpoint['model']

        # remove "backbone." from weights
        for key, weight in checkpoint_weights.items():
            new_key = key.replace('model.', '')
            new_key = new_key.replace('backbone.', '')
            state_dict[new_key] = weight

        # remove keys of final fully-connected layer
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')

        if 'resnet' in name:
            if n_input_channels == 1:
                # sum the weights of the first convolution for depth, same as
                # triple input
                state_dict['conv1.weight'] = torch.sum(
                    state_dict['conv1.weight'],
                    axis=1,
                    keepdim=True
                )
            elif n_input_channels == 4:
                # append the sum of the weights for rgb for the fourth channel
                # divide by 2 to keep the same magnitude
                state_dict['conv1.weight'] = torch.cat([
                    state_dict['conv1.weight'],
                    torch.sum(state_dict['conv1.weight'], axis=1, keepdim=True),
                ], axis=1)
                state_dict['conv1.weight'] /= 2

        # Multimodal swin transformer only need
        default_embedder = 'patch_embedder.0.weight' in state_dict
        if default_embedder:
            dim = state_dict['patch_embedder.0.weight'].shape[1]
            default_embedder = default_embedder and dim == 3

        if 'swin' in name and 'multi' in name and default_embedder:
            old_weight = state_dict.pop('patch_embedder.0.weight')
            old_bias = state_dict.pop('patch_embedder.0.bias')
            rgb_dim = backbone.patch_embedder.rgb_embed_dim
            if n_input_channels == 4:
                state_dict['patch_embedder.rgb_layers.0.weight'] = old_weight[:rgb_dim, :3]
                state_dict['patch_embedder.rgb_layers.0.bias'] = old_bias[:rgb_dim]

            state_dict['patch_embedder.depth_layers.0.weight'] = old_weight.sum(axis=1, keepdim=True)[rgb_dim:]
            state_dict['patch_embedder.depth_layers.0.bias'] = old_bias[rgb_dim:]

            old_ln_weight = state_dict.pop('patch_embedder.2.weight')
            old_ln_bias = state_dict.pop('patch_embedder.2.bias')
            state_dict['patch_embedder.rgb_layers.2.weight'] = old_ln_weight[:rgb_dim]
            state_dict['patch_embedder.rgb_layers.2.bias'] = old_ln_bias[:rgb_dim]
            state_dict['patch_embedder.depth_layers.2.weight'] = old_ln_weight[rgb_dim:]
            state_dict['patch_embedder.depth_layers.2.bias'] = old_ln_bias[rgb_dim:]

        elif 'swin' in name and 'multi' not in name:
            dim = state_dict['patch_embedder.0.weight'].shape[1]
            old_weight = state_dict.pop('patch_embedder.0.weight')
            weights_sum = old_weight.sum(axis=1, keepdim=True)
            if dim == n_input_channels:
                state_dict['patch_embedder.0.weight'] = old_weight
            elif n_input_channels == 1:
                state_dict['patch_embedder.0.weight'] = weights_sum
            elif n_input_channels == 4:
                state_dict['patch_embedder.0.weight'] = torch.cat(
                    [old_weight, weights_sum], axis=1
                )
                state_dict['patch_embedder.0.weight'] /= 2

        backbone.load_state_dict(state_dict, strict=True)

    return backbone
