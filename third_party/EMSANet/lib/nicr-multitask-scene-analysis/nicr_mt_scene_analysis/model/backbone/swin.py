# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Union, List, Type

import warnings

import torch
from torch import nn
from torchvision.models import swin_transformer
from torchvision.ops import Permute

from nicr_mt_scene_analysis.model.backbone import Backbone
from nicr_mt_scene_analysis.model.normalization import get_normalization_class


class SwinBackbone(Backbone):

    def __init__(self,
                 model: swin_transformer.SwinTransformer,
                 n_input_channels: int = 3):
        # This class is just a small wrapper around the torchvision swin transformer.
        # It enables to extract features from the different stages of the swin transformer.
        # Additionally it removes the classification head from the original imagenet model.
        # It should be noted, that the features after each stage are outputted in NHWC format,
        # as this is the format which the next stage expects.
        # Also before using the features from the skip connection, a additional layer norm should be applied.
        # Only the features from the last stage are outputted in NCHW format.
        super().__init__()
        # get the dim of the patch embedding
        embed_dim = int(model.features[0][0].out_channels)

        # Create a custom patch embedder for other than for 3 input channels
        with torch.no_grad():
            if n_input_channels != 3:
                embedder = nn.Conv2d(n_input_channels,
                                     embed_dim,
                                     kernel_size=model.features[0][0].kernel_size,
                                     stride=model.features[0][0].stride)

                # copy weights from original embedder to new one.
                weights = model.features[0][0].state_dict()
                old_weight = weights['weight']
                weights_sum = old_weight.sum(axis=1, keepdim=True)
                # if the input has only one channel, the weights can be summed
                if n_input_channels == 1:
                    weights['weight'] = weights_sum
                # If the input has 4 channels, the original weights can be kept
                # and for the new channel the weights can be summed.
                # To keep the weights in the same range, the weights are divided by 2.
                elif n_input_channels == 4:
                    weights_append = [weights_sum]
                    weights_append.insert(0, old_weight)
                    weights_new = torch.cat(weights_append, axis=1)
                    weights_new /= 2
                    weights['weight'] = weights_new
                else:
                    raise ValueError(f"Unknown number of input channels: {n_input_channels}")

                embedder.load_state_dict(weights)
                model.features[0][0] = embedder

        self.patch_embedder = model.features[0]
        # stage 1 only consists of swin modules
        self.layer_1 = model.features[1]
        # all other stages consist of patch merging and swin modules
        self.layer_2 = model.features[2:4]
        self.layer_3 = model.features[4:6]

        # Swin outputs NHWC but the rest of the architecture required
        # NCHW, this is why the permute is required.
        # The ouput of every other stage gets permuted in the
        # encoder-decoder fusion.
        self.layer_4 = nn.Sequential(
            model.features[6:8],
            model.norm,
            Permute((0, 3, 1, 2))
        )

        # define stages and meta information
        self._stages = [
            self.patch_embedder,
            self.layer_1,
            self.layer_2,
            self.layer_3,
            self.layer_4
        ]
        self._stages_n_channels = [
            embed_dim,
            embed_dim,
            embed_dim*2,
            embed_dim*4,
            embed_dim*8
        ]
        self._stages_downsampling = [
            4,
            4,
            8,
            16,
            32
        ]
        self._stages_memory_layout = [
            'nhwc',
            'nhwc',
            'nhwc',
            'nhwc',
            'nchw'
        ]

    @property
    def stages(self) -> List[Union[torch.nn.Sequential, torch.nn.Module]]:
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


def get_swin_backbone(
    name: str,
    pretrained_torchvision: bool,
    n_input_channels: int,
    normalization: Type[nn.Module] = get_normalization_class('ln'),
    **kwargs: Any
) -> SwinBackbone:
    name = name.lower()
    model = None
    weights = None

    is_default = True
    swin_embed_dims = {
        'swin-t': 96,
        'swin-s': 96,
        'swin-b': 128,
        'swin-t-v2': 96,
        'swin-s-v2': 96,
        'swin-b-v2': 128,
        'swin-t-128': 128,
        'swin-t-v2-128': 128
    }

    embed_dim = swin_embed_dims[name]

    # set correct weights for torchvision swin transformer implementation
    if pretrained_torchvision and not is_default:
        warnings.warn(
            'The pretrained weights from torchvision are only available for '
            'the default embedding dimensions of the swin transformer models. '
            'The model will be initialized randomly.'
        )
    if pretrained_torchvision and is_default:
        weights = 'IMAGENET1K_V1'

    # Because we want to enable small modifications to the swin transformer
    # (e.g. the embedding dimension, number of heads), we have to create the
    # model manually.
    progress = True
    model = None

    # Load weights if available
    if 'swin-t' in name:
        if '-v2' in name:
            weights = swin_transformer.Swin_V2_T_Weights.verify(weights)
        else:
            weights = swin_transformer.Swin_T_Weights.verify(weights)

    elif 'swin-s' in name:
        if '-v2' in name:
            weights = swin_transformer.Swin_V2_S_Weights.verify(weights)
        else:
            weights = swin_transformer.Swin_S_Weights.verify(weights)

    elif 'swin-b' in name:
        if '-v2' in name:
            weights = swin_transformer.Swin_V2_B_Weights.verify(weights)
        else:
            weights = swin_transformer.Swin_B_Weights.verify(weights)

    # setup common arguments for all swin models
    args_swin_common = {
        'patch_size': [4, 4],
        'weights': weights,
        'progress': progress,
        'norm_layer': normalization,
    }

    args_swin = {
        # swin v1 uses 7x7 windows
        'window_size': [7, 7],
    }

    args_swin_v2 = {
        # swin v2 uses 8x8 windows
        'window_size': [8, 8],
        # Also use the newer swin transformer block and patch merging in v2
        'block': swin_transformer.SwinTransformerBlockV2,
        'downsample_layer': swin_transformer.PatchMergingV2,
    }

    # Select the correct arguments for the swin model
    if '-v2' in name:
        args_swin_common.update(args_swin_v2)
    else:
        args_swin_common.update(args_swin)

    depth = {
        'swin-t': [2, 2, 6, 2],
        'swin-s': [2, 2, 18, 2],
        'swin-b': [2, 2, 18, 2],
        'swin-t-v2': [2, 2, 6, 2],
        'swin-s-v2': [2, 2, 18, 2],
        'swin-b-v2': [2, 2, 18, 2],
        'swin-t-128': [2, 2, 6, 2],
        'swin-t-v2-128': [2, 2, 6, 2]
    }
    args_swin_common['depths'] = depth[name]

    num_heads = {
        'swin-t': [3, 6, 12, 24],
        'swin-s': [3, 6, 12, 24],
        'swin-b': [4, 8, 16, 32],
        'swin-t-v2': [3, 6, 12, 24],
        'swin-s-v2': [3, 6, 12, 24],
        'swin-b-v2': [4, 8, 16, 32],
        'swin-t-128': [4, 8, 16, 32],
        'swin-t-v2-128': [4, 8, 16, 32]
    }
    args_swin_common['num_heads'] = num_heads[name]

    args_swin_common['embed_dim'] = embed_dim
    stochastic_depth_prob = {
        'swin-t': 0.2,
        'swin-s': 0.3,
        'swin-b': 0.5,
        'swin-t-v2': 0.2,
        'swin-s-v2': 0.3,
        'swin-b-v2': 0.5,
        'swin-t-128': 0.2,
        'swin-t-v2-128': 0.2
    }
    args_swin_common['stochastic_depth_prob'] = stochastic_depth_prob[name]

    args_swin_common.update(kwargs)

    model = swin_transformer._swin_transformer(**args_swin_common)

    # Wrap model in backbone which supports the feature extractor
    model = SwinBackbone(model,
                         n_input_channels=n_input_channels)

    return model
