# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple

import torch
import numpy as np

from ...data.preprocessing.resize import get_fullres_key
from ...data.preprocessing.resize import get_fullres_shape
from ...types import BatchType
from ...types import DecoderRawOutputType
from ...types import PostprocessingOutputType
from .dense_base import DensePostprocessingBase
from .instance import InstancePostprocessing
from ...utils.panoptic_merge import deeplab_merge_batch
from .semantic import SemanticPostprocessing


class PanopticPostprocessing(DensePostprocessingBase):
    def __init__(
        self,
        semantic_postprocessing: SemanticPostprocessing,
        instance_postprocessing: InstancePostprocessing,
        semantic_classes_is_thing: Tuple[bool],
        semantic_class_has_orientation: Tuple[bool],
        normalized_offset: bool = True,
        compute_scores: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        # store preprocessing objects
        self._semantic_postprocessing = semantic_postprocessing
        self._instance_postprocessing = instance_postprocessing

        # convert list of booleans to list of thing class indices
        self._thing_class_ids = np.where(semantic_classes_is_thing)[0]

        # +1 cause void class is missing in semantic_classes_is_thing
        self._thing_ids_panoptic = np.where(semantic_classes_is_thing)[0]+1
        self._orientation_ids = np.where(semantic_class_has_orientation)[0]+1

        self._normalized_offset = normalized_offset

        self._compute_scores = compute_scores

        self._max_instances_per_category = 1 << 16

    @property
    def max_instances_per_category(self):
        return self._max_instances_per_category

    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs of panoptic helper
        (s_output, i_output), (s_side_outputs, i_side_outputs) = data

        # just call both postprocessing methods
        r_dict_semantic = self._semantic_postprocessing._postprocess_training(
            data=(s_output, s_side_outputs),
            batch=batch
        )
        r_dict_instance = self._instance_postprocessing._postprocess_training(
            data=(i_output, i_side_outputs),
            batch=batch
        )

        return {**r_dict_semantic, **r_dict_instance}

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs of panoptic helper
        (s_output, i_output), (s_side_outputs, i_side_outputs) = data

        # just call both postprocessing methods
        r_dict_semantic = self._semantic_postprocessing._postprocess_inference(
            data=(s_output, s_side_outputs),
            batch=batch
        )
        r_dict_instance = self._instance_postprocessing._postprocess_inference(
            data=(i_output, i_side_outputs),
            batch=batch
        )
        r_dict = {**r_dict_semantic, **r_dict_instance}

        # add additional postprocessing for panoptic segmentation
        # unpack instance decoder outputs
        with_orientation = (3 == len(i_output))
        if with_orientation:
            center_heatmap, center_offset, orientation = i_output
        else:
            center_heatmap, center_offset = i_output

        # undo instance offset normalizing if applied
        if self._normalized_offset:
            h, w = center_offset.shape[-2:]
            center_offset_ = center_offset.detach().clone()
            center_offset_[:, 0] = center_offset_[:, 0] * h
            center_offset_[:, 1] = center_offset_[:, 1] * w
        else:
            center_offset_ = center_offset

        # note that we compute the foreground mask before resizing to full
        # resolution
        # moreover, note that this mask does may not be valid for masking the
        # resulting panoptic segmentation, as panoptic merging may change
        # classes or instances
        semantic_segmentation = r_dict['semantic_segmentation_idx']
        foreground_mask = torch.isin(
            semantic_segmentation,
            test_elements=torch.tensor(self._thing_class_ids,
                                       device=semantic_segmentation.device)
        )
        r_dict['panoptic_foreground_mask'] = foreground_mask

        # get instance segmentation with panoptic foreground mask
        post = self._instance_postprocessing
        instance_segmentation, instance_segmentation_meta = \
            post._get_instance_segmentation(center_heatmap,
                                            center_offset_,
                                            foreground_mask)
        # r_dict['panoptic_instance_segmentation'] = instance_segmentation
        # r_dict['panoptic_instance_segmentation_meta'] = \
        #     instance_segmentation_meta

        # create panoptic segmentation using deeplab merging, putting more
        # focus on instances and assign the semantic class based on
        # majority vote
        # we perform merging on cpu as it is faster, all panoptic stuff is on
        # cpu
        panoptic_seg, panoptic_ids = deeplab_merge_batch(
            (semantic_segmentation.cpu() + 1),    # +1 because void is missing
            instance_segmentation.cpu(),
            foreground_mask.cpu(),
            self._max_instances_per_category,
            self._thing_ids_panoptic,
            0
        )
        r_dict['panoptic_segmentation_deeplab'] = panoptic_seg

        # store matched panoptic ids for orientation matching later
        r_dict['panoptic_segmentation_deeplab_ids'] = panoptic_ids

        # derive semantic from panoptic segmentation (with void)
        # note that deeplab merging may have changed the semantic classes
        pan_seg_semantic = panoptic_seg // self._max_instances_per_category
        r_dict['panoptic_segmentation_deeplab_semantic_idx'] = pan_seg_semantic

        # note, we use the raw instance segmentation from above as the instance
        # ids are unique, after merging, the ids are enumerated for each
        # semantic class and, thus, not unique anymore
        r_dict['panoptic_segmentation_deeplab_instance_idx'] = instance_segmentation
        r_dict['panoptic_segmentation_deeplab_instance_meta'] = instance_segmentation_meta

        # update panoptic instance segmentation meta dict, i.e., add semantic
        # and compute dense scores for semantic, instance, and panoptic
        if self._compute_scores:
            # compute semantic score (semantic may have changed)
            semantic_scores = r_dict['semantic_softmax_scores']
            pan_semantic_idx = pan_seg_semantic.to(semantic_scores.device)
            pan_semantic_idx = pan_semantic_idx.unsqueeze(1)    # add channel axis
            # note that we have pay attention to void, as it has no valid score
            void_mask = pan_semantic_idx == 0
            # void handling 1: simply select class 0 scores for void labels
            pan_semantic_idx -= 1    # account for void
            pan_semantic_idx[void_mask] = 0
            # extract scores
            pan_seg_semantic_score = torch.take_along_dim(
                semantic_scores,
                pan_semantic_idx,
                dim=1
            )
            # void handling 2: assign score 0 to void pixels
            pan_seg_semantic_score[void_mask] = 0.
            # remove temporary channel axis and force cpu
            pan_seg_semantic_score = pan_seg_semantic_score.squeeze(1).cpu()

            r_dict['panoptic_segmentation_deeplab_semantic_score'] = pan_seg_semantic_score

            # instance and panoptic scores
            # panoptic score is computed similar to YOLO and PanopticDeeplab:
            # - stuff classes: semantic score
            # - thing classes: instance score * mean semantic score of instance
            pan_seg_instance_score = torch.zeros_like(panoptic_seg,
                                                      dtype=torch.float32)
            # copy semantic score to panoptic score (for stuff classes)
            pan_seg_panoptic_score = torch.clone(pan_seg_semantic_score)
            for batch_idx in range(pan_seg_instance_score.shape[0]):
                # note that we have to map panoptic ids back to the instance ids
                # before merging to get the instance scores
                for pan_id, ins_id in panoptic_ids[batch_idx].items():
                    # create mask for current panoptic id
                    mask = panoptic_seg[batch_idx] == pan_id

                    # get instance score and copy to result
                    instance_score = instance_segmentation_meta[batch_idx][ins_id]['score']
                    pan_seg_instance_score[batch_idx][mask] = instance_score

                    # compute semantic score of an instance as mean of all
                    # semantic scores
                    semantic_score = torch.mean(
                        pan_seg_semantic_score[batch_idx][mask]
                    )

                    instance_segmentation_meta[batch_idx][ins_id]['semantic_score'] = semantic_score.item()
                    instance_segmentation_meta[batch_idx][ins_id]['semantic_idx'] = semantic_score.item()

                    # add semantic class as well (use first element, all
                    # elements should be equal)
                    semantic_idx = pan_seg_semantic[batch_idx][mask][0]
                    instance_segmentation_meta[batch_idx][ins_id]['semantic_idx'] = semantic_idx.item()

                    # compute panoptic score as semantic*instance
                    panoptic_score = semantic_score*instance_score
                    pan_seg_panoptic_score[batch_idx][mask] = panoptic_score
                    instance_segmentation_meta[batch_idx][ins_id]['panoptic_score'] = panoptic_score.item()

                    # add panoptic id to instance meta
                    instance_segmentation_meta[batch_idx][ins_id]['panoptic_id'] = pan_id

            r_dict['panoptic_segmentation_deeplab_instance_score'] = pan_seg_instance_score
            r_dict['panoptic_segmentation_deeplab_panoptic_score'] = pan_seg_panoptic_score

        # resize output to original shape (assume same shape for all samples)
        shape = get_fullres_shape(batch, 'instance')

        r_dict[get_fullres_key('panoptic_segmentation_deeplab')] = \
            self._resize_prediction(
                panoptic_seg,
                shape=shape,
                mode='nearest'
            )

        r_dict[get_fullres_key('panoptic_segmentation_deeplab_instance_idx')] = \
            self._resize_prediction(
                r_dict['panoptic_segmentation_deeplab_instance_idx'],
                shape=shape,
                mode='nearest'
            )

        r_dict[get_fullres_key('panoptic_segmentation_deeplab_semantic_idx')] = \
            self._resize_prediction(
                r_dict['panoptic_segmentation_deeplab_semantic_idx'],
                shape=shape,
                mode='nearest'
            )

        if self._compute_scores:
            r_dict[get_fullres_key('panoptic_segmentation_deeplab_semantic_score')] = \
                self._resize_prediction(
                    pan_seg_semantic_score,
                    shape=shape,
                    mode='nearest'
                )
            r_dict[get_fullres_key('panoptic_segmentation_deeplab_instance_score')] = \
                self._resize_prediction(
                    pan_seg_instance_score,
                    shape=shape,
                    mode='nearest'
                )
            r_dict[get_fullres_key('panoptic_segmentation_deeplab_panoptic_score')] = \
                self._resize_prediction(
                    pan_seg_panoptic_score,
                    shape=shape,
                    mode='nearest'
                )

        # orientation estimation
        if with_orientation:
            # orientation fg mask
            foreground_mask_orientation = torch.isin(
                pan_seg_semantic,
                test_elements=torch.tensor(self._orientation_ids,
                                           device=pan_seg_semantic.device)
            )

            r_dict['orientations_panoptic_segmentation_deeplab_instance'] = \
                post._get_instance_orientation(
                    orientation=orientation,
                    instance_segmentation=r_dict['panoptic_segmentation_deeplab_instance_idx'],
                    foreground_mask=foreground_mask_orientation
                )

        return r_dict
