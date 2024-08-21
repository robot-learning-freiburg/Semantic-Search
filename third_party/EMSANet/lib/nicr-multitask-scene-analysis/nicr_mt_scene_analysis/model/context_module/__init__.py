# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Soehnke Benedikt Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Tuple, Type, Union

from torch import nn
from torch import Tensor

from ..activation import get_activation_class
from ..normalization import get_normalization_class

from .appm import AdaptivePyramidPoolingModule
from .ppm import PyramidPoolingModule
from .none import NoContextModule


KNOWN_CONTEXT_MODULES = (
    # pyramid pooling module:
    # -> fixed output sizes (h=w)
    # -> adaptive pooling window sizes, if input resolution changes
    'ppm',    # default, same as ppm-1-5
    'ppm-1-5',    # output sizes: 1, 5 (for 640x480 inputs)
    'ppm-1-5-10',    # output sizes: 1, 5, 10 (for 1280x960 inputs)
    'ppm-1-2-4-8',    # output sizes: 1, 2, 4, 8 (1024x512 inputs)
    # adaptive pyramid pooling module:
    # -> adaptive output sizes (h=w), if input resolution changes (e.g. during
    #    validation on fullres images of Cityscapes)
    # -> fixed pooling window sizes
    'appm',    # same as ppm, but with adaption
    'appm-1-5',    # same as ppm-1-5, but with adaption
    'appm-1-5-10',    # same as ppm-1-5-10, but with adaption
    'appm-1-2-4-8',    # same as ppm-1-2-4-8, but with adaption
    # no context module at all
    'none'
)


ContextModuleType = Union[PyramidPoolingModule,
                          AdaptivePyramidPoolingModule,
                          NoContextModule]


def get_context_module(
    name: str,
    n_channels_in: int,
    n_channels_out: int,
    input_size: Tuple[int, int],
    normalization: str = 'batchnorm',
    activation: str = 'relu',
    upsampling: str = 'bilinear'
) -> ContextModuleType:
    name = name.lower()

    if name not in KNOWN_CONTEXT_MODULES:
        raise ValueError(f"Unknown context module: '{name}'")

    if 'appm' in name:
        if 'appm-1-2-4-8' == name:
            bins = (1, 2, 4, 8)
        elif 'appm-1-5-10' == name:
            bins = (1, 5, 10)
        else:
            bins = (1, 5)
        context_module_class = AdaptivePyramidPoolingModule
    elif 'ppm' in name:
        if 'ppm-1-2-4-8' == name:
            bins = (1, 2, 4, 8)
        elif 'ppm-1-5-10' == name:
            bins = (1, 5, 10)
        else:
            bins = (1, 5)
        context_module_class = PyramidPoolingModule
    elif 'none' == name:
        bins = ()
        context_module_class = NoContextModule

    context_module = context_module_class(
        n_channels_in, n_channels_out,
        bins=bins,
        input_size=input_size,
        normalization=get_normalization_class(normalization),
        activation=get_activation_class(activation),
        upsampling=upsampling
    )

    return context_module
