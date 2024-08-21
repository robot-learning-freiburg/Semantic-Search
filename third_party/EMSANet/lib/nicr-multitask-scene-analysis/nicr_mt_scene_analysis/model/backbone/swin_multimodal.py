# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Callable, List, Type, Dict

import warnings

import torch
from torch import nn
from torchvision.ops import Permute
from torchvision.models import swin_transformer

from nicr_mt_scene_analysis.model.normalization import get_normalization_class
from .swin import SwinBackbone


class MergedPatchEmbedder(nn.Module):
    def __init__(self, rgb_embed_dim: int, depth_embed_dim: int,
                 patch_size: List[int], norm_layer: nn.Module):
        super().__init__()
        self.rgb_embed_dim = rgb_embed_dim
        self.depth_embed_dim = depth_embed_dim

        # Each modality has its own patch embedder. The structure is the same
        # as for standard swin transformer.
        self.rgb_layers = nn.Sequential(
            nn.Conv2d(
                3, rgb_embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
            ),
            Permute([0, 2, 3, 1]),
            norm_layer(rgb_embed_dim),
        )

        self.depth_layers = nn.Sequential(
            nn.Conv2d(
                1, depth_embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
            ),
            Permute([0, 2, 3, 1]),
            norm_layer(depth_embed_dim),
        )

    def update_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # old_weight = state_dict.pop('patch_embedder.0.weight')
        # old_bias = state_dict.pop('patch_embedder.0.bias')

        # state_dict['patch_embedder.rgb_layers.0.weight'] = old_weight[:self.rgb_embed_dim, :3]
        # state_dict['patch_embedder.rgb_layers.0.bias'] = old_bias[:self.rgb_embed_dim]
        # if old_weight.shape[1] == 3:
        #     state_dict['patch_embedder.depth_layers.0.weight'] = old_weight.sum(axis=1, keepdim=True)[self.rgb_embed_dim:]
        # else:
        #     state_dict['patch_embedder.depth_layers.0.weight'] = old_weight[self.rgb_embed_dim:, 3:]
        # state_dict['patch_embedder.depth_layers.0.bias'] = old_bias[self.rgb_embed_dim:]

        # old_ln_weight = state_dict.pop('patch_embedder.2.weight')
        # old_ln_bias = state_dict.pop('patch_embedder.2.bias')

        # state_dict['patch_embedder.rgb_layers.2.weight'] = old_ln_weight[:self.rgb_embed_dim]
        # state_dict['patch_embedder.rgb_layers.2.bias'] = old_ln_bias[:self.rgb_embed_dim]
        # state_dict['patch_embedder.depth_layers.2.weight'] = old_ln_weight[self.rgb_embed_dim:]
        # state_dict['patch_embedder.depth_layers.2.bias'] = old_ln_bias[self.rgb_embed_dim:]

        return state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In the codebase each backbone is expected to only get one input.
        # This is because for FusedEncoder each backbone will just get its
        # modality as input.
        # For handling both modalities in one backbones the input (rgb, depth)
        # will just get concatenated along the channel dimension. This means
        # that the input will have shape (batch_size, 4, height, width).
        # The first three channels are the rgb channels and the last channel
        # is the depth channel.
        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :]
        # We use two separate patch embedder for rgb and depth so the
        # two modalities don't get mixed up that early in the model.
        rgb_features = self.rgb_layers(rgb)
        depth_features = self.depth_layers(depth)
        # Each backbone is also just expected to return one output.
        # Because of this we concatenate the two modalities along the
        # channel dimension. In this case its the last dimension, because
        # after the patch embedder the swin transformer expects the the input
        # to be in NHWC format.
        return torch.cat([rgb_features, depth_features], dim=3)


class SwinMultimodalBackbone(SwinBackbone):
    def __init__(self, model: swin_transformer.SwinTransformer,
                 rgb_embed_dim: int, depth_embed_dim: int,
                 patch_size: List[int], norm_layer: nn.Module,
                 **kwargs: Any):
        # Enforce 3 input channels because the patch embedder will
        # get replaced anyway.
        super().__init__(model=model, n_input_channels=3, **kwargs)
        self.patch_embedder = MergedPatchEmbedder(
            rgb_embed_dim, depth_embed_dim, patch_size, norm_layer
        )

        # Update the stages to include the new patch embedder.
        self._stages = [
            self.patch_embedder,
            self.layer_1,
            self.layer_2,
            self.layer_3,
            self.layer_4
        ]


def get_swin_multimodal_backbone(
    name: str,
    pretrained_torchvision: bool,
    n_input_channels: int,
    normalization: Type[nn.Module] = get_normalization_class('ln'),
    **kwargs: Any
) -> SwinMultimodalBackbone:
    name = name.lower()
    model = None
    weights = None

    is_default = True
    swin_embed_dims = {
        'swin-multi-t': 96,
        'swin-multi-s': 96,
        'swin-multi-b': 128,
        'swin-multi-t-v2': 96,
        'swin-multi-s-v2': 96,
        'swin-multi-b-v2': 128,
        'swin-multi-t-128': 128,
        'swin-multi-t-v2-128': 128,
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
        'swin-multi-t': [2, 2, 6, 2],
        'swin-multi-s': [2, 2, 18, 2],
        'swin-multi-b': [2, 2, 18, 2],
        'swin-multi-t-v2': [2, 2, 6, 2],
        'swin-multi-s-v2': [2, 2, 18, 2],
        'swin-multi-b-v2': [2, 2, 18, 2],
        'swin-multi-t-128': [2, 2, 6, 2],
        'swin-multi-t-v2-128': [2, 2, 6, 2]
    }
    args_swin_common['depths'] = depth[name]

    num_heads = {
        'swin-multi-t': [3, 6, 12, 24],
        'swin-multi-s': [3, 6, 12, 24],
        'swin-multi-b': [4, 8, 16, 32],
        'swin-multi-t-v2': [3, 6, 12, 24],
        'swin-multi-s-v2': [3, 6, 12, 24],
        'swin-multi-b-v2': [4, 8, 16, 32],
        'swin-multi-t-128': [4, 8, 16, 32],
        'swin-multi-t-v2-128': [4, 8, 16, 32]
    }
    args_swin_common['num_heads'] = num_heads[name]

    args_swin_common['embed_dim'] = embed_dim
    stochastic_depth_prob = {
        'swin-multi-t': 0.2,
        'swin-multi-s': 0.3,
        'swin-multi-b': 0.5,
        'swin-multi-t-v2': 0.2,
        'swin-multi-s-v2': 0.3,
        'swin-multi-b-v2': 0.5,
        'swin-multi-t-128': 0.2,
        'swin-multi-t-v2-128': 0.2
    }
    args_swin_common['stochastic_depth_prob'] = stochastic_depth_prob[name]

    args_swin_common.update(kwargs)

    model = swin_transformer._swin_transformer(**args_swin_common)

    # The usual input channels for the swin transformer are 3 (RGB) and 1 (gray).
    # This gets assert here because, currently we only can handle this config.
    assert n_input_channels == 4

    rgb_embed_dim = {
        # 96 * 2 // 3 = 64, so that the total embedding dimension is 96
        'swin-multi-t':  64,
        'swin-multi-s': 64,
        # 128 * 3 // 4 = 96, so that the total embedding dimension is 128
        'swin-multi-b': 96,
        'swin-multi-t-v2': 64,
        'swin-multi-s-v2': 64,
        'swin-multi-b-v2': 96,
        'swin-multi-t-128': 96,
        'swin-multi-t-v2-128': 96
    }

    depth_embed_dim = {
        # 96 * 1 // 3 = 32, so that the total embedding dimension is 96
        'swin-multi-t': 32,
        'swin-multi-s': 32,
        # 128 * 1 // 4 = 32, so that the total embedding dimension is 128
        'swin-multi-b': 32,
        'swin-multi-t-v2': 32,
        'swin-multi-s-v2': 32,
        'swin-multi-b-v2': 32,
        'swin-multi-t-128': 32,
        'swin-multi-t-v2-128': 32
    }

    # Wrap model in backbone which supports the feature extractor
    model = SwinMultimodalBackbone(model,
                                   rgb_embed_dim=rgb_embed_dim[name],
                                   depth_embed_dim=depth_embed_dim[name],
                                   patch_size=args_swin_common['patch_size'],
                                   norm_layer=normalization)

    return model
