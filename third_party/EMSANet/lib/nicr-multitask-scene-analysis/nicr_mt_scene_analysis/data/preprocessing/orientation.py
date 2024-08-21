# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Tuple, Union

import numpy as np

from ...types import BatchType
from ...utils import np_rad2biternion
from .multiscale_supervision import _enable_multiscale
from .utils import _keys_available
from nicr_scene_analysis_datasets.dataset_base import OrientationDict


class OrientationTargetGenerator:
    def __init__(
        self,
        semantic_classes_estimate_orientation: Union[Tuple[bool], None] = None
    ) -> None:
        if semantic_classes_estimate_orientation is not None:
            # convert list of booleans to list of class ids
            self._orientation_class_ids = \
                np.where(semantic_classes_estimate_orientation)[0]
        else:
            self._orientation_class_ids = None

    @_enable_multiscale
    def __call__(self, sample: BatchType, **kwargs: Any) -> BatchType:
        if not _keys_available(sample, ('instance', 'orientations',
                                        'semantic')):
            # might be a multiscale call with disabled multiscale for instances
            # or inference
            return sample

        # get height and width
        height, width = sample['instance'].shape

        # create orientation image as two-channel image -> HWC with C = 2
        orientation_img = np.zeros((height, width, 2), dtype='float32')
        # create empty foreground image
        foreground_img = np.zeros((height, width), dtype='bool')

        orientations_present = OrientationDict()

        for instance_id in np.unique(sample['instance']):
            if 0 == instance_id:
                # 0 indicates no instance -> skip
                continue

            if instance_id not in sample['orientations']:
                # we do not have an orientation for this instance
                continue

            # get mask
            mask = sample['instance'] == instance_id

            # check semantic label of instance
            # note, we use mode of all relevant pixels as the pixels may have
            # different labels due to instance merging from 3d boxes
            if self._orientation_class_ids is not None:
                # As the bincount always starts with 0, the argmax will always
                # return the class id of the most frequent class of the
                # semantic labels.
                semantic_class = np.bincount(
                    sample['semantic'][mask].flatten()
                ).argmax()
                if semantic_class not in self._orientation_class_ids:
                    # we are not interested in this class
                    continue

            # fill in encoded orientation and adapt foreground
            orientation_img[mask] = np_rad2biternion(
                sample['orientations'][instance_id]
            )
            foreground_img = np.logical_or(foreground_img, mask)
            orientations_present[instance_id] = \
                sample['orientations'][instance_id]

        sample['orientation'] = orientation_img
        sample['orientation_foreground'] = foreground_img
        sample['orientations_present'] = orientations_present

        return sample
