# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np

from ...types import BatchType
from .resize import resize
from .utils import _get_input_shape
from .utils import _get_relevant_spatial_keys


class RandomCrop:
    def __init__(self, crop_height: int, crop_width: int) -> None:
        self._crop_height = crop_height
        self._crop_width = crop_width

    def __call__(self, sample: BatchType) -> BatchType:
        h, w = _get_input_shape(sample)

        # resize image if it is too small
        scale = 1.0
        if h <= self._crop_height:
            # resize height to match at least crop height
            scale = max(self._crop_height / h, scale)
        if w <= self._crop_width:
            # resize width to match at least crop width
            scale = max(self._crop_width / w, scale)

        if scale > 1.0:
            h, w = int(h*scale+0.5), int(w*scale+0.5)
            sample = resize(sample, height=h, width=w)

        # determine slices
        if (h-self._crop_height) > 0:
            y_start = np.random.randint(0, h - self._crop_height)
        else:
            y_start = 0
        if (w-self._crop_width) > 0:
            x_start = np.random.randint(0, w - self._crop_width)
        else:
            x_start = 0
        slice_y = slice(y_start, y_start+self._crop_height)
        slice_x = slice(x_start, x_start+self._crop_width)

        # apply random crop
        for key in _get_relevant_spatial_keys(sample):
            sample[key] = sample[key][slice_y, slice_x, ...]

        return sample
