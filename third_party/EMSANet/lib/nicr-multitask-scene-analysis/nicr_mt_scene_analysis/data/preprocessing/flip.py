# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np

from ...types import BatchType
from .utils import _get_relevant_spatial_keys


class RandomHorizontalFlip:
    def __init__(self, p: float) -> None:
        self._p = p

    def __call__(self, sample: BatchType) -> BatchType:
        if 'orientations_present' in sample:
            raise RuntimeError("Do not apply `RandomHorizontalFlip` after "
                               "`OrientationTargetGenerator`.")

        if np.random.uniform() <= self._p:
            # spatial entries
            for key in _get_relevant_spatial_keys(sample):
                sample[key] = np.flip(sample[key], axis=1)

            if 'orientations' in sample:
                # mirror orientations at y axis
                for id_ in sample['orientations']:
                    sample['orientations'][id_] = \
                        (2*np.pi - sample['orientations'][id_]) % (2*np.pi)

        return sample
