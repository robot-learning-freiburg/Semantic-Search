# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Some parts of this code are based on:
    https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ...data.preprocessing.resize import get_fullres_key
from ...data.preprocessing.resize import get_fullres_shape
from ...utils import biternion2rad
from ...types import BatchType
from ...types import DecoderRawOutputType
from ...types import PostprocessingOutputType
from .dense_base import DensePostprocessingBase


class InstancePostprocessing(DensePostprocessingBase):
    def __init__(
        self,
        heatmap_threshold: float = 0.1,
        heatmap_nms_kernel_size: int = 3,
        heatmap_apply_foreground_mask: bool = False,
        top_k_instances: int = 64,
        normalized_offset: bool = True,
        offset_distance_threshold: Union[None, int] = None,
        **kwargs
    ) -> None:
        super().__init__()

        assert heatmap_nms_kernel_size % 2 == 1
        assert top_k_instances <= 254
        self._heatmap_nms_kernel_size = heatmap_nms_kernel_size
        self._heatmap_nms_padding = (self._heatmap_nms_kernel_size - 1) // 2
        self._heatmap_threshold = heatmap_threshold
        self._top_k_instances = top_k_instances
        self._normalized_offset = normalized_offset

        # added after EMSANet release
        self._heatmap_apply_foreground_mask = heatmap_apply_foreground_mask
        self._offset_distance_threshold = offset_distance_threshold

        if 'output_width' in kwargs and 'output_height' in kwargs:
            self._mesh_grid = self._get_mesh_grid(kwargs['output_width'],
                                                  kwargs['output_height'],
                                                  torch.device('cpu'))
        else:
            self._mesh_grid = None

        self.debug = kwargs.get('debug', False)

    def _get_mesh_grid(
        self,
        height: int,
        width: int,
        device: torch.device
    ) -> torch.Tensor:
        x = torch.arange(0, width, device=device)
        y = torch.arange(0, height, device=device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((y_grid, x_grid), dim=0)

        # add batch axis
        grid = grid.unsqueeze(0)

        return grid

    def _get_instance_centers(
        self,
        center_heatmap: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        # set values below threshold to -1 so they are not considered as
        # instance center
        center_heatmap = F.threshold(center_heatmap,
                                     self._heatmap_threshold,
                                     -1)
        # do keypoint NMS
        # note that we ignore instance centers less than (kernel size -1)/2
        # pixels away of the borders as this areas may contain resizing
        # artifacts
        center_heatmap_pooled = F.max_pool2d(
            center_heatmap,
            kernel_size=self._heatmap_nms_kernel_size,
            stride=1,
        )
        center_heatmap_pooled = F.pad(center_heatmap_pooled,
                                      pad=(self._heatmap_nms_padding,)*4)
        center_heatmap[center_heatmap != center_heatmap_pooled] = -1
        batch_size = center_heatmap.shape[0]

        # find top k instances for each batch - shape of scores is  (b, topk)
        scores, _ = torch.topk(center_heatmap.flatten(start_dim=1),
                               k=self._top_k_instances, dim=1)

        # change shape from (b, 1, h, w) to (b, h, w)
        center_heatmap = center_heatmap.squeeze_(dim=1)

        # optionally mask out predicted centers that are not part of the
        # foreground (note, applying the mask to the raw centers may split them
        # in multiple blobs and, thus, may results in additional centers)
        if self._heatmap_apply_foreground_mask:
            center_heatmap[torch.logical_not(foreground_mask)] = -1

        # find the minimum value of the top-k instances for each batch
        # additionally, unsqueeze so it is (b, 1, 1)
        lowest_top_k_value = scores[:, -1, None, None]
        # clamp top k to min=0, else it does not count as an instance center
        lowest_top_k_value = lowest_top_k_value.clamp_(min=0)

        # apply threshold
        instance_centers = [
            (center_heatmap[b_idx] >= lowest_top_k_value[b_idx])
            for b_idx in range(batch_size)
        ]

        # create a list of length b with tensors of shape (y, x) for ever
        # instance center location for each element in the batch
        # note:
        # - integer casting is done to save memory
        # - due to this the maximum coordinate of a pixel coordinate is 2^31-1,
        #   which should be more than enough
        # - uint16 would be even better cause negative pixel coordinates are
        #   not possible, however, PyTorch does not support uint16 yet
        instance_center_list = [centers.nonzero().int()
                                for centers in instance_centers]

        return torch.stack(instance_centers, dim=0), instance_center_list

    def _get_instance_segmentation(
        self,
        center_heatmap: torch.Tensor,
        center_offset: torch.Tensor,
        foreground_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get instance centers
        _, instance_centers_list = self._get_instance_centers(center_heatmap,
                                                              foreground_mask)

        # create required mesh grid lazily
        device = center_offset.device
        if self._mesh_grid is None or self._mesh_grid.device != device:
            _, _, h, w = center_offset.shape
            self._mesh_grid = self._get_mesh_grid(h, w, device)

        # unpack shape
        b, c, h, w = center_offset.shape

        # repeat mesh grid at batch axis
        mesh_grid = self._mesh_grid.expand(b, c, h, w)

        # calculate the position of pixels after applying offset vectors
        location = mesh_grid + center_offset

        # flat spatial dimensions and change shape
        # (b, 2, h, w) to (b, 2, h*w) to (b, h*w, 2)
        location = location.flatten(start_dim=2).permute(0, 2, 1)

        # flat spatial dimensions (b, h, w) to (b, h*w)
        flat_foreground_mask = foreground_mask.flatten(start_dim=1)

        # prepare result of instance segmentation (same shape as foreground
        # mask - (b, h, w) - and same device)
        instance_segmentation = torch.zeros_like(foreground_mask,
                                                 dtype=torch.uint8)

        # prepare additional list of dictionaries storing some meta
        # information for the instances in each example of the batch
        instance_meta = [{} for _ in range(b)]

        for idx, centers in enumerate(instance_centers_list):
            # skip if no instances were detected
            if centers.shape[0] == 0:
                continue

            # create a flat view of the instance segmentation of current batch
            instance_view = instance_segmentation[idx].view(-1)

            # use foreground mask to only consider pixels that are a thing
            loc = location[idx]
            foreground = flat_foreground_mask[idx]
            loc = loc[foreground]

            # reshape to (1, h*w, 2)
            loc = loc.unsqueeze(0)
            # reshape to (n, 1, 2)
            centers = centers.unsqueeze(1)

            # find the minimum distance for each pixel
            distance = torch.norm(centers - loc, dim=-1)

            # determine instance id based on minimum distance and offset by 1
            # to avoid stuff id 0
            min_distances, instance_id = torch.min(distance, dim=0)
            instance_id = (instance_id + 1).type(torch.uint8)

            # optionally threshold minimum distances
            # missing instance centers or an erroneous semantic segmentation
            # (foreground) may lead to pixels being assigned to instance centers
            # quite far away
            # for application, it might be useful to mask such pixels out by
            # assigning the no instance label (id=0)
            # note that this masking leads to thing segments without an
            # instance id, which have to be handled later on
            if self._offset_distance_threshold is not None:
                instance_id[min_distances > self._offset_distance_threshold] = 0
                # print((instance_id == 0).sum())

            instance_view[foreground] = instance_id

            # fill meta dictionary
            instance_areas = torch.bincount(instance_id,
                                            minlength=len(centers)+1)
            for i, center in enumerate(centers, start=1):    # 0 = no instance
                center = center[0]   # remove added dimension
                y = center[0].item()
                x = center[1].item()
                # note that the area and the center coordinates are derived
                # from the raw prediction, the network output might be further
                # umsampled later
                instance_meta[idx][i] = {
                    'center_yx': (y, x),
                    'area': instance_areas[i].item(),
                    'score': center_heatmap[idx, 0, y, x].item()
                }

        return instance_segmentation, instance_meta

    def _get_instance_orientation(
        self,
        orientation: torch.Tensor,
        instance_segmentation: torch.Tensor,
        foreground_mask: torch.Tensor
    ) -> List[Dict[int, float]]:
        batch_size = instance_segmentation.shape[0]

        # create empty result list
        results = []

        # get orientations for each sample in the batch independently as
        # instance ids are only unique within a single sample
        for idx in range(batch_size):
            results_dict = {}
            # get flat views for segmentation, orientation, and foreground
            # (b, h, w) -> (h*w)
            flat_segmentation = instance_segmentation[idx].flatten()
            # (b, c=2, h, w) -> (c=2, h, w) -> (c=2, h*w)
            flat_orientation = orientation[idx].flatten(start_dim=1)

            if foreground_mask is not None:
                # (b, h, w) -> (h, w) -> (h*w)
                flat_foreground_mask = foreground_mask[idx].flatten()

                # mask background
                flat_segmentation = flat_segmentation[flat_foreground_mask]
                flat_orientation = flat_orientation[:, flat_foreground_mask]

            # get averaged orientation for each instance id
            for instance_id in torch.unique(flat_segmentation):
                if instance_id == 0:
                    # 0 is no instance
                    continue

                mask = flat_segmentation == instance_id

                # average sin and cos parts to get average angle (note output
                # normalization ensures unit length)
                avg_orientation = flat_orientation[:, mask].sum(dim=1)

                # add batch axis and convert to rad
                avg_angle = biternion2rad(avg_orientation[None, :])[0]

                # we return python types int and float
                results_dict[instance_id.item()] = avg_angle.item()
            results.append(results_dict)

        return results

    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs
        output, side_outputs = data

        # create results dict
        r_dict = {
            'instance_output': output,
            'instance_side_outputs': side_outputs
        }

        return r_dict

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack decoder outputs (there are no side outputs)
        output, side_outputs = data

        with_orientation = (3 == len(output))
        if with_orientation:
            center_heatmap, center_offset, orientation = output
        else:
            center_heatmap, center_offset = output

        r_dict = {
            'instance_output': output,
            'instance_side_outputs': side_outputs,
            'instance_centers': center_heatmap,
            'instance_offsets': center_offset
        }
        if with_orientation:
            r_dict['instance_orientation'] = orientation

        # undo normalizing if applied
        if self._normalized_offset:
            h, w = center_offset.shape[-2:]
            center_offset_ = center_offset.detach().clone()
            center_offset_[:, 0] = center_offset_[:, 0] * h
            center_offset_[:, 1] = center_offset_[:, 1] * w
        else:
            center_offset_ = center_offset

        # get instance segmentation
        # i-1: using gt foreground mask (dataset evaluation)
        if 'instance_foreground' in batch:
            foreground_mask = batch['instance_foreground']
            segmentation, meta = self._get_instance_segmentation(
                center_heatmap=center_heatmap,
                center_offset=center_offset_,
                foreground_mask=foreground_mask
            )
            # used to visualize predicted centers
            r_dict['instance_segmentation_gt_foreground'] = segmentation
            # note that the area and the center coordinates are derived
            # from the raw prediction, the network output might be further
            # umsampled later
            r_dict['instance_segmentation_gt_meta'] = meta

            # resize to original shape (assume same shape for all samples)
            # note: we resize only the final prediction as resizing centers and
            # offsets is much more complicated
            shape = get_fullres_shape(batch, 'instance')
            r_dict[get_fullres_key('instance_segmentation_gt_foreground')] = \
                self._resize_prediction(
                    segmentation,
                    shape=shape,
                    mode='nearest'
                )

        # i-2: considering everything as foreground (for debugging)
        if self.debug:
            foreground_mask = torch.ones_like(center_heatmap, dtype=torch.bool)
            segmentation, _ = self._get_instance_segmentation(
                center_heatmap=center_heatmap,
                center_offset=center_offset_,
                foreground_mask=foreground_mask
            )
            r_dict['instance_segmentation_all_foreground'] = segmentation

            # resize to original shape (assume same shape for all samples)
            # note: we resize only the final prediction as resizing centers and
            # offsets is much more complicated
            shape = get_fullres_shape(batch, 'instance')
            r_dict[get_fullres_key('instance_segmentation_all_foreground')] = \
                self._resize_prediction(
                    segmentation,
                    shape=shape,
                    mode='nearest'
                )

        if not with_orientation:
            # we do not need orientation postprocessing, stop here and return
            return r_dict

        # additionally get instance orientation
        # note: we do not resize to original shape before as it does not change
        # the results much and orientations need to be normalized

        # o-1: for instances based on gt instance segmentation and gt
        #      foreground mask for known orientations (dataset evaluation)
        if all(k in batch for k in ('instance', 'orientation_foreground')):
            r_dict['orientations_gt_instance_gt_orientation_foreground'] = \
                self._get_instance_orientation(
                    orientation=orientation,
                    instance_segmentation=batch['instance'],
                    foreground_mask=batch['orientation_foreground']
                )

        # o-2: for instances based on predicted instance segmentation and gt
        #      foreground mask for known orientations (dataset evaluation)
        if all(k in batch for k in ['instance_foreground',
                                    'orientation_foreground']):
            r_dict['orientations_instance_segmentation_gt_orientation_foreground'] = \
                self._get_instance_orientation(
                    orientation=orientation,
                    instance_segmentation=r_dict['instance_segmentation_gt_foreground'],
                    foreground_mask=batch['orientation_foreground']
                )

        if self.debug:
            # o-3: for all instances based on gt instance segmentation (for
            #      debugging)
            r_dict['orientations_gt_instance'] = \
                self._get_instance_orientation(
                    orientation=orientation,
                    instance_segmentation=batch['instance'],
                    foreground_mask=None
                )
            # o-4: for all instances based on predicted instance segmentation
            r_dict['orientations_instance_segmentation'] = \
                self._get_instance_orientation(
                    orientation=orientation,
                    instance_segmentation=r_dict['instance_segmentation_gt_foreground'],
                    foreground_mask=None
                )

        return r_dict
