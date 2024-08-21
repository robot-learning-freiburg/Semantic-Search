# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Type

from torch import nn

from ...types import ContextModuleInputType
from ...types import ContextModuleOutputType
from ..activation import get_activation_class
from ..normalization import get_normalization_class
from ..utils import ConvNormAct


class NoContextModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        **kwargs: Any
    ) -> None:
        super().__init__()
        if n_channels_out != n_channels_in:
            self.layer = ConvNormAct(n_channels_in, n_channels_out,
                                     kernel_size=1,
                                     normalization=normalization,
                                     activation=activation)
        else:
            self.layer = nn.Identity()

        # we do not have a reduction, we simply apply a conv+bn+act to
        # adapt the number of channels, we assign n_channels_out to
        # n_channels_reduction to ensure compatibility in scene decoder
        self.n_channels_reduction = n_channels_out

    def forward(self, x: ContextModuleInputType) -> ContextModuleOutputType:
        x = self.layer(x)
        return x, ()
