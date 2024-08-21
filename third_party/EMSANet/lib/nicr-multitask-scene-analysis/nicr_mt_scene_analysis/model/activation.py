# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Optional, Type

from torch import nn

from ..utils import partial_class


KNOWN_ACTIVATIONS = [
    'relu',
    'silu', 'swish'
]


def get_activation_class(
    name: Optional[str] = None,
    **kwargs: Any
) -> Type[nn.Module]:
    # global default
    if name is None:
        name = 'relu'
        kwargs['inplace'] = True

    name = name.lower()
    if name not in KNOWN_ACTIVATIONS:
        raise ValueError(f"Unknown activation: '{name}'")

    if 'relu' == name:
        cls = nn.ReLU
    elif name in ['swish', 'silu']:
        cls = nn.SiLU

    return partial_class(cls, **kwargs)
