# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Tuple

import numpy as np

from ...data import CollateIgnoredDict
from ...types import BatchType
from ...utils.panoptic_merge import naive_merge_semantic_and_instance_np
from .utils import _keys_available


class PanopticTargetGenerator:
    def __init__(
        self,
        semantic_classes_is_thing: Tuple[bool],    # with void
    ) -> None:
        # convert list of booleans to list of thing class indices
        self._thing_class_ids = np.where(semantic_classes_is_thing)[0]
        # hypersim has more than 256 instances per image
        self._max_instances_per_category = (1 << 16)
        self._void_label = 0

    def __call__(self, sample: BatchType) -> BatchType:
        if not _keys_available(sample, ('instance', 'semantic')):
            # might be an inference call
            return sample

        semantic = sample['semantic']
        instance = sample['instance']

        panoptic_targets, panoptic_targets_id_dicts = \
            naive_merge_semantic_and_instance_np(
                semantic,
                instance,
                max_instances_per_category=self._max_instances_per_category,
                thing_ids=self._thing_class_ids,
                void_label=self._void_label
            )

        sample['panoptic'] = panoptic_targets
        # used for instance matching
        sample['panoptic_ids_to_instance_dict'] = \
            CollateIgnoredDict(panoptic_targets_id_dicts)

        return sample
