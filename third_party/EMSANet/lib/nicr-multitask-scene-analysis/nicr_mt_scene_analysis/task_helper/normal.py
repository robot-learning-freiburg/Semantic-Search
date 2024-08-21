# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Tuple

import torch

from ..data.preprocessing.resize import get_fullres
from ..data.preprocessing.resize import get_fullres_key
from ..loss import MSELoss
from ..loss import L1Loss
from ..metric import RootMeanSquaredError
from ..types import BatchType
from ..visualization import visualize_normal_pil
from .base import append_detached_losses_to_logs
from .base import append_profile_to_logs
from .base import TaskHelperBase


KNOWN_NORMAL_LOSS_FUNCTIONS = (
    'mse',
    'l1'
)


class NormalTaskHelper(TaskHelperBase):
    def __init__(
        self,
        loss_name: str,
        disable_multiscale_supervision: bool = False
    ) -> None:
        super().__init__()
        assert loss_name in KNOWN_NORMAL_LOSS_FUNCTIONS
        if 'mse' == loss_name:
            self._loss_class = MSELoss
        elif 'l1' == loss_name:
            self._loss_class = L1Loss

        self._disable_multiscale_supervision = disable_multiscale_supervision

        # during validation, we store some examples for visualization purposes
        self._examples = {}

    def initialize(self, device: torch.device):
        # loss
        self._loss = self._loss_class(
            reduction='sum'    # see _compute_loss before changing this!
        )
        # metric
        self._metric_rmse = RootMeanSquaredError().to(device)

    def _compute_losses(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:
        # collect predictions and targets for loss
        no_multiscale = self._disable_multiscale_supervision
        preds, targets, keys = self.collect_predictions_and_targets_for_loss(
            batch=batch,
            batch_key='normal',
            predictions_post=predictions_post,
            predictions_post_key='normal_output',
            side_outputs_key=None if no_multiscale else 'normal_side_outputs'
        )

        # get mask of valid pixels for each scale, i.e. pixels with gt normal
        masks_valid = [_get_valid_gt_normals(gt) for gt in targets]
        # count number of valid pixels for each scale
        n_elements_valid = [mask.sum().cpu().detach().item()
                            for mask in masks_valid]

        # set invalid predictions to zero to ensure that the loss for these
        # pixels is 0 and does not affect the reduced (summed) loss
        preds_masked = []
        for mask, pred in zip(masks_valid, preds):
            # add additional channel axis and expand to same shape as input by
            # repeating at channel axis
            mask = mask.unsqueeze(1).expand_as(pred)
            preds_masked.append(pred*mask)

        # compute losses
        loss_outputs = self._loss(input_tensors=preds_masked,
                                  target_tensors=targets)

        # create log dict
        loss_dict = {
            f'normal_loss_{key}': loss/n
            for key, (loss, _), n in zip(keys, loss_outputs, n_elements_valid)
        }

        # compute total loss (ignore total counts of the loss output object)
        total_loss = self.accumulate_losses(
            losses=[loss for loss, _ in loss_outputs],
            n_elements=n_elements_valid
        )

        # append unweighted total loss (for advanced multi-task learning)
        loss_dict[self.mark_as_total('normal')] = total_loss

        return loss_dict

    @append_profile_to_logs('normal_step_time')
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

    @append_profile_to_logs('normal_step_time')
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

        # update metric
        target = get_fullres(batch, 'normal')
        mask = _get_valid_gt_normals(target)
        self._metric_rmse.update(
            preds=predictions_post[get_fullres_key('normal_output')],
            target=target,
            mask=mask
        )

        # store example for visualization (not fullres!)
        if batch_idx == 0:
            ex = predictions_post['normal_output'][0].cpu().numpy()
            ex = ex.transpose(1, 2, 0)    # chw -> hwc
            key = f'normal_example_batch_{batch_idx}_0'
            self._examples[key] = visualize_normal_pil(ex)

        return loss_dict, {}

    @append_profile_to_logs('normal_epoch_end_time')
    def validation_epoch_end(self):
        artifacts = {}
        logs = {'normal_rmse': self._metric_rmse.compute()}

        # reset metric (it is not done automatically)
        self._metric_rmse.reset()

        return artifacts, self._examples, logs


def _get_valid_gt_normals(gt):
    invalid = (gt[:, 0, ...] == 0) & (gt[:, 1, ...] == 0) & (gt[:, 2, ...] == 0)
    return ~invalid
