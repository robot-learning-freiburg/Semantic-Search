# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Parts of this code are taken and adapted from:
    https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/data/transforms/target_transforms.py
"""
from typing import Any, Dict, Tuple, Union

import numpy as np

from ...types import BatchType
from ...data.preprocessing.resize import get_fullres
from .multiscale_supervision import _enable_multiscale
from .utils import _keys_available


class InstanceClearStuffIDs:
    def __init__(
        self,
        semantic_classes_is_thing: Tuple[bool],    # with void
    ) -> None:
        # get stuff class ids without void
        semantic_classes_is_stuff = np.logical_not(semantic_classes_is_thing)
        self._stuff_class_ids = np.where(semantic_classes_is_stuff)[0]
        self._stuff_class_ids = self._stuff_class_ids    # including void

    def __call__(self, sample: BatchType) -> BatchType:
        if not _keys_available(sample, ('instance', 'semantic')):
            # might be an inference call
            return sample

        # depending on the dataset and the applied division into stuff and
        # thing classes, the data may contain valid ids for instances of stuff
        # classes, we force id=0 (= no instance) for all stuff classes
        # including void to ensure that each stuff class is considered as a
        # single segment later

        # note that this preprocessor should be applied before resizing to
        # ensure that this requirement also applies to the full resolution
        # images that may be used for determining evaluation metrics

        # get mask of stuff classes
        stuff_mask = np.isin(sample['semantic'], self._stuff_class_ids)

        # force id=0 inplace
        sample['instance'][stuff_mask] = 0

        return sample


class InstanceTargetGenerator:
    def __init__(
        self,
        sigma: int,
        semantic_classes_is_thing: Union[Tuple[bool], None] = None,  # with void
        sigma_for_additional_downscales: Union[Dict[int, int], None] = None,
        normalized_offset: bool = True
    ) -> None:
        self._sigma_for_downscales = {None: sigma}
        if sigma_for_additional_downscales is not None:
            self._sigma_for_downscales = {**self._sigma_for_downscales,
                                          **sigma_for_additional_downscales}

        # store some precomputed stuff
        self._2d_gauss_for_downscale = {
            scale: self._precompute_2d_gauss(sigma)
            for scale, sigma in self._sigma_for_downscales.items()
        }
        self._mesh_grid_for_downscale = {}    # created lazily later

        if semantic_classes_is_thing is not None:
            # convert list of booleans to list of thing class indices
            self._thing_class_ids = np.where(semantic_classes_is_thing)[0]
            semantic_classes_is_stuff = np.logical_not(semantic_classes_is_thing)
            self._stuff_class_ids = np.where(semantic_classes_is_stuff)[0]
            self._stuff_class_ids = self._stuff_class_ids[1:]    # remove void
        else:
            self._thing_class_ids = None
            self._stuff_class_ids = None

        self._normalized_offset = normalized_offset

    @staticmethod
    def _precompute_2d_gauss(sigma):
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1

        return np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    @_enable_multiscale
    def __call__(
        self,
        sample: BatchType,
        downscale=None,
        **kwargs: Any
    ) -> BatchType:
        if 'instance' not in sample:
            # might be a multiscale call with disabled multiscale for instances
            # or inference
            return sample

        # extract shape and instance image
        instance_image = sample['instance']
        height, width = instance_image.shape

        # get mesh grid for offset encoding or create it lazily
        if downscale not in self._mesh_grid_for_downscale:
            grid = np.meshgrid(range(height), range(width), indexing='ij')
            self._mesh_grid_for_downscale[downscale] = grid
        else:
            grid = self._mesh_grid_for_downscale[downscale]

        # get gauss for scale
        gauss = self._2d_gauss_for_downscale[downscale]

        # encode instances
        foreground = np.zeros((height, width), dtype='bool')
        center_img = np.zeros((height, width), dtype='float32')
        offset_img = np.zeros((height, width, 2), dtype='int16')

        for instance_id in np.unique(instance_image):
            if 0 == instance_id:
                # 0 indicates no instance -> skip
                continue

            # get mask for instance
            mask_indices = np.where(instance_image == instance_id)

            # check semantic label of instance
            # note, we use mode of all relevant pixels as the pixels may have
            # different labels due to instance merging from 3d boxes
            if self._thing_class_ids is not None:
                # As the bincount always starts with 0, the argmax will always
                # return the class id of the most frequent class of the
                # semantic labels.
                semantic_class = np.bincount(
                    sample['semantic'][mask_indices]
                ).argmax()
                if semantic_class not in self._thing_class_ids:
                    # we are not interested in this class
                    continue

            # add instance to foreground
            foreground[mask_indices] = True

            # encode instance center
            sigma = self._sigma_for_downscales[downscale]
            center_y, center_x = np.mean(mask_indices, axis=1)
            center_y, center_x = int(center_y), int(center_x)
            # determine upper left point of gauss in image
            ul = (int(np.round(center_x - 3*sigma - 1)),
                  int(np.round(center_y - 3*sigma - 1)))
            # determine bottom right point of gauss in image
            br = (int(np.round(center_x + 3*sigma + 2)),
                  int(np.round(center_y + 3*sigma + 2)))
            # determine slices in gauss pattern and image
            s_gauss_x = slice(max(0, -ul[0]), min(br[0], width) - ul[0])
            s_gauss_y = slice(max(0, -ul[1]), min(br[1], height) - ul[1])
            s_img_x = slice(max(0, ul[0]), min(br[0], width))
            s_img_y = slice(max(0, ul[1]), min(br[1], height))

            # add center encoding to image
            center_img[s_img_y, s_img_x] = np.maximum(
                center_img[s_img_y, s_img_x],
                gauss[s_gauss_y, s_gauss_x]
            )

            # encode instance offset
            offset_img[mask_indices] = np.stack(
                (center_y - grid[0][mask_indices],
                 center_x - grid[1][mask_indices]),
                axis=-1
            )

        if self._normalized_offset:
            # normalize offsets to [0, 1], requires converting to float here
            offset_img = offset_img.astype('float32')
            offset_img[..., 0] /= offset_img.shape[0]
            offset_img[..., 1] /= offset_img.shape[1]

        sample['instance_center'] = center_img
        sample['instance_offset'] = offset_img
        sample['instance_foreground'] = foreground

        # force that all stuff classes have instance id 0 (0 = no instance)
        # if you face an assert here, consider adding InstanceClearStuffIDs
        assert (instance_image[~foreground] == 0).all()
        instance_fullres = get_fullres(sample, 'instance')
        if instance_fullres is not None:
            semantic_fullres = get_fullres(sample, 'semantic')
            instance_foreground = np.isin(semantic_fullres,
                                          self._thing_class_ids)
            assert (instance_fullres[~instance_foreground] == 0).all()

        # create separate mask for instance centers
        sample['instance_center_mask'] = sample['instance_foreground'].copy()
        if self._stuff_class_ids is not None:
            # we have semantic information, adapt mask for instance centers
            # such that predicted centers in stuff regions are penalized
            stuff_foreground = np.isin(sample['semantic'],
                                       self._stuff_class_ids)
            sample['instance_center_mask'][stuff_foreground] = True

        return sample
