# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple

import numpy as np

from ...types import BatchType
from .utils import _keys_available


class SemanticClassMapper:
    def __init__(
        self,
        classes_to_map: Tuple[int],
        new_label: int = 0
    ) -> None:
        self._semantic_classes_to_map = np.array(classes_to_map)
        self._new_label = new_label

    def __call__(self, sample: BatchType) -> BatchType:
        if not _keys_available(sample, ('semantic',)):
            # might be an inference call
            return sample

        # get mask for all classes that should be mapped
        mask = np.isin(sample['semantic'],
                       self._semantic_classes_to_map)

        # apply mapping inplace
        sample['semantic'][mask] = self._new_label

        return sample
