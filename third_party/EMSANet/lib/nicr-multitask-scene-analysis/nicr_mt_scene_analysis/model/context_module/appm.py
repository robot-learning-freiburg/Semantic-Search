# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...types import ContextModuleInputType
from ...types import ContextModuleOutputType
from ..activation import get_activation_class
from ..normalization import get_normalization_class
from ..utils import ConvNormAct


class AdaptivePyramidPoolingModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        input_size: Tuple[int, int],
        bins: Tuple[int, ...] = (1, 2, 3, 6),
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: str = 'bilinear'
    ) -> None:
        super().__init__()

        n_channels_reduction = n_channels_in // len(bins)
        self._upsampling = upsampling
        self._input_size = input_size
        self._bins = bins

        features = []
        for _ in bins:
            features.append(nn.Sequential(
                nn.Identity(),    # to make ppm and appm interchangeable
                ConvNormAct(n_channels_in, n_channels_reduction,
                            kernel_size=1,
                            normalization=normalization,
                            activation=activation)
            ))
        self.features = nn.ModuleList(features)

        n_channels_in_last_conv = n_channels_in + n_channels_reduction*len(bins)
        self.final_conv = ConvNormAct(n_channels_in_last_conv, n_channels_out,
                                      kernel_size=1,
                                      normalization=normalization,
                                      activation=activation)
        self.n_channels_reduction = n_channels_reduction

    def forward(self, x: ContextModuleInputType) -> ContextModuleOutputType:
        h, w = x.shape[2:]
        h_inp, w_inp = self._input_size

        # adapt bins to current size of x
        bin_multiplier_h = int((h / h_inp) + 0.5)
        bin_multiplier_w = int((w / w_inp) + 0.5)

        out = [x]
        features_context = []
        for f, bin_ in zip(self.features, self._bins):
            h_pool = bin_ * bin_multiplier_h
            w_pool = bin_ * bin_multiplier_w
            pooled = F.adaptive_avg_pool2d(x, (h_pool, w_pool))
            y = f(pooled)
            features_context.append(y)
            if 'nearest' == self._upsampling:
                out.append(F.interpolate(y, (int(h), int(w)),
                                         mode='nearest'))
            elif 'bilinear' == self._upsampling:
                out.append(F.interpolate(y, (int(h), int(w)),
                                         mode='bilinear',
                                         align_corners=False))
            else:
                raise NotImplementedError()

        out = torch.cat(out, 1)
        out = self.final_conv(out)

        return out, tuple(features_context)
