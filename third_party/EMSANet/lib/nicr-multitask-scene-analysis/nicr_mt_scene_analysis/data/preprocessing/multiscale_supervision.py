# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import re
from typing import Tuple

from functools import wraps

from ...types import BatchType
from .clone import clone_entries
from .resize import resize
from .utils import _keys_available
from .utils import _get_input_shape


MULTI_DOWNSCALE_KEY_FMT = '_down_{}'


def _enable_multiscale(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # apply normal preprocessing
        sample = f(*args, **kwargs)

        # multiscale
        if len(args) == 2:
            # self and sample are given as args
            args = (args[0], )

        for key in sample:
            res = re.match(MULTI_DOWNSCALE_KEY_FMT.format('([0-9]+)'), key)
            if res is None:
                # key does not match
                continue

            # apply preprocessing to multiscale entry
            kwargs['sample'] = sample[key]
            kwargs['downscale'] = int(res.groups()[0])
            sample[key] = f(*args, **kwargs)

        return sample

    return wrapper


def get_downscale(sample: BatchType, downscale) -> BatchType:
    return sample.get(MULTI_DOWNSCALE_KEY_FMT.format(downscale), None)


class MultiscaleSupervisionGenerator:
    def __init__(
        self,
        downscales: Tuple[int],
        keys: Tuple[str]
    ) -> None:
        self._downscales = downscales
        self._keys = keys

    @property
    def downscales(self):
        return self._downscales

    def __call__(self, sample: BatchType) -> BatchType:
        if not _keys_available(sample, self._keys):
            raise KeyError(f"At least one key of '{self._keys}' is missing in"
                           "`sample`.")

        # read shape from rgb or depth image (at least one is available)
        h, w = _get_input_shape(sample)

        for downscale in self._downscales:
            # clone selected entries
            cloned_sample = clone_entries(sample, keys_to_clone=self._keys)

            # resize
            cloned_sample = resize(cloned_sample,
                                   height=int(h / downscale),
                                   width=int(w / downscale))

            # add resized multiscale clone to dict
            key = MULTI_DOWNSCALE_KEY_FMT.format(downscale)
            sample[key] = cloned_sample

        return sample
