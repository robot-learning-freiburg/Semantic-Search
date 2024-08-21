# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Optional, Type

from torch import nn

from ..utils import partial_class


KNOWN_NORMALIZATIONS = (
    'bn', 'batchnorm',
    'ln', 'layernorm'
)


def get_normalization_class(
    name: Optional[str] = None,
    **kwargs: Any
) -> Type[nn.Module]:
    # global default
    name = name or 'bn'

    name = name.lower()
    if name not in KNOWN_NORMALIZATIONS:
        raise ValueError(f"Unknown normalization: '{name}'")

    if name in ('bn', 'batchnorm'):
        cls = nn.BatchNorm2d

    if name in ('ln', 'layernorm'):
        cls = nn.LayerNorm

    return partial_class(cls, **kwargs)
