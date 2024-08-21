# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Tuple

import torch
import numpy as np

from ..data.preprocessing.resize import get_fullres
from ..data.preprocessing.resize import get_fullres_key
from ..loss import L1Loss
from ..loss import MSELoss
from ..loss import VonMisesLossBiternion
from ..metric.mae import MeanAbsoluteAngularError
from ..metric.mae import PanopticQualityWithOrientationMAE
from ..utils.panoptic_merge import deeplab_merge_batch
from ..types import BatchType
from ..visualization import visualize_instance_center_pil
from ..visualization import visualize_instance_offset_pil
from ..visualization import visualize_instance_pil
from ..visualization import visualize_instance_orientations_pil
from ..visualization import visualize_orientation_pil
from .base import append_detached_losses_to_logs
from .base import append_profile_to_logs
from .base import TaskHelperBase


KNOWN_INSTANCE_CENTER_LOSS_FUNCTIONS = (
    'mse',
    'l1'
)


class InstanceTaskHelper(TaskHelperBase):
    def __init__(
        self,
        semantic_n_classes: int,
        semantic_classes_is_thing: Tuple[bool],
        loss_name_instance_center: str = 'mse',
        disable_multiscale_supervision: bool = False
    ) -> None:
        super().__init__()

        self._loss_name_instance_center = loss_name_instance_center
        self._disable_multiscale_supervision = disable_multiscale_supervision

        # required for panoptic quality
        self._semantic_n_classes = semantic_n_classes
        self._semantic_classes_is_thing = semantic_classes_is_thing

        self._with_orientation = False     # will be detected later on the fly

        # during validation, we store some examples for visualization purposes
        self._examples = {}

        # hypersim has more than 256 instances per image
        self._max_instances_per_category = 1 << 16     # 1 << 8
        self._thing_ids = np.where(self._semantic_classes_is_thing)[0]

    def initialize(self, device: torch.device):
        # losses
        assert self._loss_name_instance_center in KNOWN_INSTANCE_CENTER_LOSS_FUNCTIONS
        if 'mse' == self._loss_name_instance_center:
            self._loss_center = MSELoss(
                reduction='sum'    # see _compute_loss before changing this!
            )
        else:
            self._loss_center = L1Loss(
                reduction='sum'    # see _compute_loss before changing this!
            )
        self._loss_offset = L1Loss(
            reduction='sum'    # see _compute_loss before changing this!
        )
        self._loss_orientation = VonMisesLossBiternion()

        # create instance of metrics
        # Note: Average Precision could also be added, but is not really of
        # interest for us, moreover it is relatively slow so it increases the
        # training time a lot, we use PQ to assess quality of the instances.
        # Further details are explained in the validation_step.
        self._mae_pq_deeplab = PanopticQualityWithOrientationMAE(
            num_categories=self._semantic_n_classes,
            ignored_label=0,
            max_instances_per_category=self._max_instances_per_category,
            offset=256**3,
            is_thing=self._semantic_classes_is_thing
        )

        self._mae_gt = MeanAbsoluteAngularError()

    def _compute_losses(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:

        # collect predictions
        no_multiscale = self._disable_multiscale_supervision
        preds, keys, downscales = self.collect_predictions_for_loss(
            predictions_post=predictions_post,
            predictions_post_key='instance_output',
            side_outputs_key=None if no_multiscale else 'instance_side_outputs'
        )
        # split predictions
        preds_center, preds_offset, preds_orientation = [], [], []
        for pred in preds:
            preds_center.append(pred[0][:, 0])    # remove channel axis
            preds_offset.append(pred[1])
            if len(pred) == 3:
                preds_orientation.append(pred[2])
        self._with_orientation = len(preds_orientation) > 0

        # compute losses for instance
        # compute losses for instance centers
        targets_center_mask = self.collect_targets_for_loss(
            batch,
            batch_key='instance_center_mask',
            downscales=downscales
        )
        targets_center = self.collect_targets_for_loss(
            batch,
            batch_key='instance_center',
            downscales=downscales
        )
        # set invalid predictions to zero to ensure that the loss for these
        # pixels is 0 and does not affect the reduced (summed) loss
        preds_center_masked = [
            pred*mask
            for mask, pred in zip(targets_center_mask, preds_center)
        ]
        loss_outputs_center = self._loss_center(
            input_tensors=preds_center_masked,
            target_tensors=targets_center
        )
        # count number of valid pixels for each scale
        n_elements_center = [mask.sum().cpu().detach().item()
                             for mask in targets_center_mask]

        # compute losses for instance offsets
        targets_instance_foreground = self.collect_targets_for_loss(
            batch,
            batch_key='instance_foreground',
            downscales=downscales
        )
        targets_offset = self.collect_targets_for_loss(
            batch,
            batch_key='instance_offset',
            downscales=downscales
        )
        # set invalid predictions to zero to ensure that the loss for these
        # pixels is 0 and does not affect the reduced (summed) loss
        preds_offset_masked = []
        for mask, pred in zip(targets_instance_foreground, preds_offset):
            # add additional channel axis and expand to same shape as input
            # by repeating at channel axis
            mask = mask.unsqueeze(1).expand_as(pred)
            preds_offset_masked.append(pred*mask)
        # count number of valid pixels for each scale
        n_elements_offset = [mask.sum().cpu().detach().item()
                             for mask in targets_instance_foreground]

        loss_outputs_offset = self._loss_offset(
            input_tensors=preds_offset_masked,
            target_tensors=targets_offset
        )

        # compute losses for orientation
        if self._with_orientation:
            targets_orientation_foreground = self.collect_targets_for_loss(
                batch,
                batch_key='orientation_foreground',
                downscales=downscales
            )

            # get real orientation targets
            targets_orientation = self.collect_targets_for_loss(
                batch,
                batch_key='orientation',
                downscales=downscales
            )

            # mask predictions and targets
            preds_orientation_masked = []
            targets_orientation_masked = []
            n_elements_orientation = []
            for pred, target, mask in zip(preds_orientation,
                                          targets_orientation,
                                          targets_orientation_foreground):
                # split in instance head creates view which is not contiguous
                pred = pred.contiguous()

                # (b, c, h, w) -> (b, h, w, c) -> (b*h*w, c) with c = 2
                pred = pred.permute((0, 2, 3, 1)).reshape(-1, 2)
                target = target.permute((0, 2, 3, 1)).reshape(-1, 2)

                # flat mask for indexing
                mask = mask.flatten()

                # count number of valid pixels
                # note: this count might be zero if there is no object with
                # orientation in current batch, to avoid division by zero
                # during loss reduction later, we set to count to 1, the loss
                # is zero anyway
                n_elements_orientation.append(
                    max(mask.sum().cpu().detach().item(), 1)
                )

                preds_orientation_masked.append(pred[mask, :])
                targets_orientation_masked.append(target[mask, :])

            # note: if there are no valid pixels, a loss of zero is returned
            loss_outputs_orientation = self._loss_orientation(
                input_tensors=preds_orientation_masked,
                target_tensors=targets_orientation_masked
            )

        # create log dict
        loss_dict = {}
        # center
        loss_dict.update({
            f'instance_center_loss_{key}': loss/n
            for key, (loss, _), n in zip(keys, loss_outputs_center,
                                         n_elements_center)
        })
        # offsets
        loss_dict.update({
            f'instance_offset_loss_{key}': loss/n
            for key, (loss, _), n in zip(keys, loss_outputs_offset,
                                         n_elements_offset)
        })

        if self._with_orientation:
            loss_dict.update({
                f'instance_orientation_loss_{key}': loss/n
                for key, (loss, _), n in zip(keys, loss_outputs_orientation,
                                             n_elements_orientation)
            })

        # compute total loss (ignore total counts of the loss output objects
        # for offsets and orientations as both apply a foreground mask)
        total_loss_center = self.accumulate_losses(
            losses=[loss for loss, _ in loss_outputs_center],
            n_elements=n_elements_center
        )
        total_loss_offset = self.accumulate_losses(
            losses=[loss for loss, _ in loss_outputs_offset],
            n_elements=n_elements_offset
        )

        if self._with_orientation:
            total_loss_orientation = self.accumulate_losses(
                losses=[loss for loss, _ in loss_outputs_orientation],
                n_elements=n_elements_orientation
            )

        # append unweighted total losses (for advanced multi-task learning)
        loss_dict.update({
            self.mark_as_total('instance_center'): total_loss_center,
            self.mark_as_total('instance_offset'): total_loss_offset,
        })

        if self._with_orientation:
            loss_dict.update({
                self.mark_as_total('instance_orientation'): total_loss_orientation,
            })

        return loss_dict

    @append_profile_to_logs('instance_step_time')
    @append_detached_losses_to_logs()
    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # compute loss
        loss_dict = self._compute_losses(
            batch=batch,
            batch_idx=batch_idx,
            predictions_post=predictions_post
        )
        return loss_dict, {}

    @append_profile_to_logs('instance_step_time')
    @append_detached_losses_to_logs()
    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # compute loss
        loss_dict = self._compute_losses(
            batch=batch,
            batch_idx=batch_idx,
            predictions_post=predictions_post
        )

        # orientation
        if self._with_orientation:
            orientations_results = predictions_post['orientations_instance_segmentation_gt_orientation_foreground']
            orientations_full_gt = predictions_post['orientations_gt_instance_gt_orientation_foreground']
            orientations_targets = batch['orientations_present']

            # update gt mae
            self._mae_gt.update(orientations_full_gt, orientations_targets)
        else:
            orientations_results = None
            orientations_targets = None

        # For evaluating the quality of the instance segmentation, without
        # semantic labels, we use ground truth data for also evaluating the
        # PQ/RQ.
        # This is useful for comparing the single task performance vs. the
        # performance of the instance task helper in the multi task setup.
        # Usually a AP computation would be a better metric for instance
        # segmentation. However, we are not directly interested in only using
        # the instance segmentation stand alone, which is why we compute the PQ
        # by using the ground truth semantic segmentation for merging.
        # Note that we also use the ground truth foreground mask, as we
        # want evaluate how close our predicted segmentation is compared
        # to the ground truth.
        # This way the computed metric gives a impression how good the instance
        # segmentation is, when we would have a perfect semantic segmentation.
        # The resulting PQ/RQ gives us a good value for analysing how good
        # the instance really is.
        semantic_batch = get_fullres(batch, 'semantic').cpu()
        instance_batch = get_fullres(batch, 'instance').cpu()
        instance_result = predictions_post[get_fullres_key('instance_segmentation_gt_foreground')].cpu()
        instance_foreground = instance_batch != 0

        # prepare the target panoptic segmentation
        panoptic_targets = get_fullres(batch, 'panoptic').cpu()
        panoptic_targets_id_dicts = batch['panoptic_ids_to_instance_dict']

        # panoptic segmentation as done in panoptic deeplab.
        # The ground truth semantic segmentation is used because we only
        # want to evaluate the instance segmentation itself.
        panoptic_deeplab_preds, panoptic_deeplab_id_dicts = \
            deeplab_merge_batch(semantic_batch, instance_result,
                                instance_foreground,
                                self._max_instances_per_category,
                                self._thing_ids,
                                0)
        panoptic_deeplab_preds = panoptic_deeplab_preds.cpu()

        self._mae_pq_deeplab.update(
            panoptic_deeplab_preds,
            orientations_results,
            panoptic_deeplab_id_dicts,
            panoptic_targets,
            orientations_targets,
            panoptic_targets_id_dicts
        )

        # store example for visualization (not fullres!)
        if batch_idx == 0:
            center, offset, *orientation = predictions_post['instance_output']

            # instances
            # raw center heatmap
            ex = center[0].cpu().numpy()[0]     # remove channel axis
            key = f'instance_center_heatmap_example_batch_{batch_idx}_0'
            self._examples[key] = visualize_instance_center_pil(
                center_img=ex,
                min_=0,
                max_=1
            )

            # raw offset
            ex = offset[0].cpu().numpy().transpose(1, 2, 0)    # chw -> hwc
            key = f'instance_offset_example_batch_{batch_idx}_0'
            self._examples[key] = visualize_instance_offset_pil(ex)

            # predicted centers (after nms, thresholding, and top-k)
            ex_meta = predictions_post['instance_segmentation_gt_meta'][0]
            key = f'instance_predicted_centers_example_batch_{batch_idx}_0'
            self._examples[key] = visualize_instance_center_pil(
                centers=tuple(i['center_yx'] for i in ex_meta.values()),
                height=center.shape[-2],
                width=center.shape[-1],
            )

            # final instances
            ex = predictions_post['instance_segmentation_gt_foreground'].cpu().numpy()[0]
            key = f'instance_instance_example_batch_{batch_idx}_0'
            self._examples[key] = visualize_instance_pil(ex)

            # orientations
            if self._with_orientation:
                # raw orientation
                orientation, = orientation    # unpack
                ex = orientation[0].cpu().numpy().transpose(1, 2, 0)    # chw -> hwc
                key = f'orientation_example_batch_{batch_idx}_0'
                self._examples[key] = visualize_orientation_pil(ex)

                # instances with orientation
                instance = batch['instance'].cpu().numpy()[0]
                orientations = predictions_post['orientations_gt_instance_gt_orientation_foreground'][0]
                key = f'instance_orientation_example_batch_{batch_idx}_0'
                self._examples[key] = visualize_instance_orientations_pil(
                    instance, orientations
                )

        return loss_dict, {}

    @append_profile_to_logs('instance_epoch_end_time')
    def validation_epoch_end(self):
        artifacts = {}
        logs = {}

        # panoptic segmentation as done in panoptic deeplab
        pq_deeplab_result = self._mae_pq_deeplab.compute(suffix="_deeplab")

        for key, value in pq_deeplab_result.items():
            if value.numel() == 1:
                logs[f'instance_{key}'] = value
            else:
                artifacts[f'instance_{key}'] = value

        # reset metric (it is not done automatically)
        self._mae_pq_deeplab.reset()

        # orientation
        if self._with_orientation:
            mae_gt_rad, mae_gt_deg = self._mae_gt.compute()
            logs['orientation_mae_gt_rad'] = mae_gt_rad
            logs['orientation_mae_gt_deg'] = mae_gt_deg

            # reset metric (it is not done automatically)
            self._mae_gt.reset()

        return artifacts, self._examples, logs
