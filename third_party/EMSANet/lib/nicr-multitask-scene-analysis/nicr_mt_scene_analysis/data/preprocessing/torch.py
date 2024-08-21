# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np
import torch

from ...types import BatchType
from .utils import _get_relevant_spatial_keys
from .multiscale_supervision import _enable_multiscale


class ToTorchTensors:
    @_enable_multiscale
    def __call__(self, sample: BatchType, **kwargs) -> BatchType:
        for key in _get_relevant_spatial_keys(sample):
            value = sample[key]

            if 3 == value.ndim:
                # (multiple) channel(s) with channels_last: HWC -> CHW
                value = value.transpose((2, 0, 1))
            elif 2 == value.ndim:
                if 'depth' == key:
                    # only one channel without channel axis: HW -> CHW
                    value = value[np.newaxis, ...]
                # otherwise nothing to do, we do not want to add channel axis
                # for masks etc.
            else:
                raise ValueError(f"Cannot handle entry '{key}' with "
                                 f"shape '{value.shape}'")

            if value.dtype == 'uint16':
                # pytorch does not support uint16
                continue

            # ensure that array is C contiguous
            value = np.ascontiguousarray(value)

            sample[key] = torch.from_numpy(value)

        return sample
