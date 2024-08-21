# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple, Union

import numpy as np
import torch

from ...types import BatchType
from .clone import DEFAULT_CLONE_KEY


def _keys_available(sample: BatchType, keys: Tuple[str]) -> bool:
    for key in keys:
        if key not in sample:
            return False
    return True


def _get_input_shape(sample: BatchType):
    """Read shape from rgb or depth image (at least one is available)"""
    if 'rgb' in sample:
        h, w, _ = sample['rgb'].shape
    else:
        h, w = sample['depth'].shape

    return h, w


def _get_relevant_spatial_keys(
    sample: BatchType,
    keys_to_ignore: Union[Tuple[str], None] = (DEFAULT_CLONE_KEY, )
) -> BatchType:
    """Filter keys for preprocessing modules"""
    keys = []
    for key, value in sample.items():
        if keys_to_ignore is not None and key in keys_to_ignore:
            # skip specified entries
            continue

        if not isinstance(value, (np.ndarray, torch.Tensor)):
            # skip primitive types such as int, str, or dict
            continue

        # we are interested in this key
        keys.append(key)

    return keys
