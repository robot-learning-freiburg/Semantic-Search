# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..data.preprocessing.resize import get_fullres
from ..data.preprocessing.resize import get_fullres_key
from ..metric.mae import PanopticQualityWithOrientationMAE
from ..metric import MeanIntersectionOverUnion
from ..types import BatchType
from .base import append_profile_to_logs
from .base import TaskHelperBase
from ..visualization import PanopticColorGenerator
from ..visualization import visualize_panoptic_pil
from ..visualization import visualize_instance_pil
from ..visualization import visualize_semantic_pil
from ..visualization import visualize_heatmap_pil

from nicr_scene_analysis_datasets.dataset_base import SemanticLabelList


class PanopticTaskHelper(TaskHelperBase):
    def __init__(
        self,
        semantic_n_classes: int,     # with void!
        semantic_classes_is_thing: Tuple[bool],
        semantic_label_list: SemanticLabelList
    ) -> None:
        super().__init__()
        # required for panoptic quality
        self._semantic_n_classes = semantic_n_classes
        self._semantic_classes_is_thing = semantic_classes_is_thing
        self._semantic_label_list = semantic_label_list

        # hypersim has more than 256 instances per image
        self._max_instances_per_category = 1 << 16     # 1 << 8
        self._thing_ids = np.where(self._semantic_classes_is_thing)[0]
        self._with_orientation = False

        # during validation, we store some examples for visualization purposes
        self._examples = {}
        self._color_generator = PanopticColorGenerator(
            classes_colors=self._semantic_label_list.colors,
            classes_is_thing=self._semantic_label_list.classes_is_thing,
            max_instances=self._max_instances_per_category,
            void_label=0,
        )

    def initialize(self, device: torch.device):
        # keep metrics on cpu
        self._mae_pq_deeplab = PanopticQualityWithOrientationMAE(
            num_categories=self._semantic_n_classes,
            ignored_label=0,
            max_instances_per_category=self._max_instances_per_category,
            offset=256**3,
            is_thing=self._semantic_classes_is_thing
        )

        # metrics (keep it on cpu, it is faster)
        self._metric_iou = MeanIntersectionOverUnion(
            n_classes=self._semantic_n_classes,     # with void!
            ignore_first_class=True     # we ignore void pixels for miou
        )
        # self._metric_iou = self._metric_iou.to(device)
        self._metric_iou.reset()

    @append_profile_to_logs('panoptic_step_time')
    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # For doing panoptic segmentation, the is no additional training_step
        # required.
        # The task helper only combines the semantic and instance segmentation
        # and calculates the PQ.
        return {}, {}

    @append_profile_to_logs('panoptic_step_time')
    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        self._with_orientation = 'orientations_present' in batch

        # get the orientation results
        if self._with_orientation:
            orientations_results = predictions_post['orientations_panoptic_segmentation_deeplab_instance']
            orientations_targets = batch['orientations_present']
        else:
            orientations_results = None
            orientations_targets = None

        # panoptic_targets the target for panoptic segmentation
        panoptic_targets = get_fullres(batch, 'panoptic').cpu()
        panoptic_targets_id_dicts = batch['panoptic_ids_to_instance_dict']

        # panoptic segmentation as done in panoptic deeplab
        panoptic_deeplab_preds = predictions_post[get_fullres_key('panoptic_segmentation_deeplab')]
        panoptic_deeplab_id_dicts = predictions_post['panoptic_segmentation_deeplab_ids']
        panoptic_deeplab_preds = panoptic_deeplab_preds.cpu()
        self._mae_pq_deeplab.update(
            panoptic_preds=panoptic_deeplab_preds,
            orientation_preds=orientations_results,
            panoptic_preds_id_dicts=panoptic_deeplab_id_dicts,
            panoptic_target=panoptic_targets,
            orientation_target=orientations_targets,
            panoptic_target_id_dicts=panoptic_targets_id_dicts
        )

        # deeplab merging can change the semantic segmentation.
        # By diving the panoptic deeplab prediction by the max instances per category,
        # we can get this semantic segmentation back.
        deeplab_semantic = (panoptic_deeplab_preds // self._max_instances_per_category)
        target_semantic = get_fullres(batch, 'semantic')
        self._metric_iou.update(preds=deeplab_semantic.cpu(),
                                target=target_semantic.cpu())

        # store example for visualization (not fullres!)
        if 0 == batch_idx:
            panoptic_seg = predictions_post['panoptic_segmentation_deeplab'][0]

            # panoptic segmentation
            key = f'panoptic_example_batch_deeplab_{batch_idx}_0'
            self._examples[key] = visualize_panoptic_pil(
                panoptic_seg.cpu().numpy(),
                shared_color_generator=self._color_generator
            )

            # semantic segmentation
            panoptic_seg_semantic = panoptic_seg // self._max_instances_per_category
            key = f'panoptic_example_batch_deeplab_semantic_{batch_idx}_0'
            self._examples[key] = visualize_semantic_pil(
                semantic_img=panoptic_seg_semantic.cpu().numpy(),
                colors=self._semantic_label_list.colors_array
            )

            # instance segmentation
            panoptic_ids = predictions_post['panoptic_segmentation_deeplab_ids'][0]
            instance_img = torch.zeros_like(panoptic_seg)
            for p_id, i_id in panoptic_ids.items():
                # map panoptic ids to instance ids (smaller numbers)
                instance_img[panoptic_seg == p_id] = i_id
            key = f'panoptic_example_batch_deeplab_instance_{batch_idx}_0'
            self._examples[key] = visualize_instance_pil(instance_img.cpu().numpy())

            # semantic score
            if 'panoptic_segmentation_deeplab_semantic_score' in predictions_post:
                ex = predictions_post['panoptic_segmentation_deeplab_semantic_score'][0]
                key = f'panoptic_example_batch_deeplab_semantic_score_{batch_idx}_0'
                self._examples[key] = visualize_heatmap_pil(
                    heatmap_img=ex.cpu().numpy(),
                    min_=0, max_=1
                )

            # instance score
            if 'panoptic_segmentation_deeplab_instance_score' in predictions_post:
                ex = predictions_post['panoptic_segmentation_deeplab_instance_score'][0]
                key = f'panoptic_example_batch_deeplab_instance_score_{batch_idx}_0'
                self._examples[key] = visualize_heatmap_pil(
                    heatmap_img=ex.cpu().numpy(),
                    min_=0, max_=1
                )
            # panoptic score
            if 'panoptic_segmentation_deeplab_panoptic_score' in predictions_post:
                ex = predictions_post['panoptic_segmentation_deeplab_panoptic_score'][0]
                key = f'panoptic_example_batch_deeplab_panoptic_score_{batch_idx}_0'
                self._examples[key] = visualize_heatmap_pil(
                    heatmap_img=ex.cpu().numpy(),
                    min_=0, max_=1
                )

        return {}, {}

    @append_profile_to_logs('panoptic_epoch_end_time')
    def validation_epoch_end(self):
        artifacts = {}
        logs = {}

        # panoptic segmentation as done in panoptic deeplab
        pq_deeplab_result = self._mae_pq_deeplab.compute(suffix="_deeplab")

        for key, value in pq_deeplab_result.items():
            if value.numel() == 1:
                logs[f'panoptic_{key}'] = value
            else:
                artifacts[f'panoptic_{key}'] = value

        # reset metric (it is not done automatically)
        self._mae_pq_deeplab.reset()

        # miou after panoptic merging as done in panoptic deeplab
        artifacts['panoptic_deeplab_semantic_cm'] = \
            self._metric_iou.confmat.clone()
        miou, ious = self._metric_iou.compute(return_ious=True)
        logs['panoptic_deeplab_semantic_miou'] = miou
        artifacts['panoptic_deeplab_semantic_ious_per_class'] = ious

        # reset metric (it is not done automatically)
        self._metric_iou.reset()
        assert self._metric_iou.confmat.sum() == 0

        return artifacts, self._examples, logs
