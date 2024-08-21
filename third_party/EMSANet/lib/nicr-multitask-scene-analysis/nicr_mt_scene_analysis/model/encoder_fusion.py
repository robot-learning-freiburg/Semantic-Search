# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Callable, Optional, Tuple, Type

from torch import Tensor
import torch.nn as nn

from ..utils import partial_class
from .activation import get_activation_class
from ..types import EncoderForwardType
from .utils import SqueezeAndExcitation


KNOWN_ENCODER_FUSIONS = (
    'se-add', 'add',    # bidirectional (note same features after first fusion)
    'add-uni-rgb', 'add-uni-depth',    # only unidirectional
    'se-add-uni-rgb', 'se-add-uni-depth',    # only unidirectional but with SE
    'none'    # do not fuse features
)


def _apply_NCHW_operation(
    x: Tensor,
    operation: Callable[[Tensor], Tensor],
    input_memory_layout: str
) -> Tensor:
    if 'nchw' == input_memory_layout:
        return operation(x)
    elif 'nhwc' == input_memory_layout:
        # it is channels last
        return operation(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    else:
        raise ValueError(f'Unknown input_memory_layout: {input_memory_layout}')


class EncoderRGBDFusionWeightedAdd(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        destinations: Tuple[str, ...],
        use_se_weighting: bool,
        input_memory_layout: str,
        activation: Type[nn.Module] = get_activation_class(),
        **kwargs
    ) -> None:
        super().__init__()

        if use_se_weighting:
            # weight features using SqueezeAndExcitation before
            self.weighting_rgb = SqueezeAndExcitation(n_channels_in,
                                                      activation=activation)
            self.weighting_depth = SqueezeAndExcitation(n_channels_in,
                                                        activation=activation)

        self._n_channels_in = n_channels_in
        self._use_se_weighting = use_se_weighting
        self._destinations = destinations
        self._input_memory_layout = input_memory_layout

    def forward(self, x: EncoderForwardType) -> EncoderForwardType:
        # unpack input
        x_rgb, x_depth = x['rgb'], x['depth']

        # apply optional weighting
        if self._use_se_weighting:
            rgb_weighted = _apply_NCHW_operation(
                x_rgb,
                self.weighting_rgb,
                self._input_memory_layout
            )
            depth_weighted = _apply_NCHW_operation(
                x_depth,
                self.weighting_depth,
                self._input_memory_layout
            )
        else:
            rgb_weighted = x_rgb
            depth_weighted = x_depth

        # fuse features
        fused = rgb_weighted + depth_weighted

        # determine where to return fused features
        y_rgb = fused if 'rgb' in self._destinations else x_rgb
        y_depth = fused if 'depth' in self._destinations else x_depth

        return {'rgb': y_rgb, 'depth': y_depth}


EncoderFusionType = EncoderRGBDFusionWeightedAdd


def get_encoder_fusion_class(
    name: Optional[str] = None,
    **kwargs: Any
) -> Type[EncoderFusionType]:
    # global default
    if name is None:
        name = 'add-uni-rgb'

    name = name.lower()
    if name not in KNOWN_ENCODER_FUSIONS:
        raise ValueError(f"Unknown encoder fusion: '{name}'")

    kwargs['use_se_weighting'] = 'se' in name

    if 'uni-rgb' in name:
        kwargs['destinations'] = ('rgb',)
    elif 'uni-depth' in name:
        kwargs['destinations'] = ('depth',)
    elif 'none' == name:
        kwargs['destinations'] = ()
    else:
        kwargs['destinations'] = ('rgb', 'depth',)

    return partial_class(EncoderRGBDFusionWeightedAdd, **kwargs)
