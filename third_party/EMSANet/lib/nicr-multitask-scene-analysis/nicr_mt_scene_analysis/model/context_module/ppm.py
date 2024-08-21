# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Parts of this code are taken and adapted from:
    https://github.com/hszhao/semseg/blob/master/model/pspnet.py
"""
from typing import Any, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...types import ContextModuleInputType
from ...types import ContextModuleOutputType
from ..activation import get_activation_class
from ..normalization import get_normalization_class
from ..utils import ConvNormAct


class PyramidPoolingModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        bins: Tuple[int, ...] = (1, 2, 3, 6),
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: str = 'bilinear',
        **kwargs: Any
    ) -> None:
        super().__init__()

        n_channels_reduction = n_channels_in // len(bins)
        self._upsampling = upsampling

        features = []
        for bin in bins:
            features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
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

        out = [x]
        features_context = []
        for f in self.features:
            y = f(x)
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
