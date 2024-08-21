# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torchmetrics import ConfusionMatrix

from ..types import BatchType
from .base import append_detached_losses_to_logs
from .base import append_profile_to_logs
from .base import TaskHelperBase


class SceneTaskHelper(TaskHelperBase):
    def __init__(
        self,
        n_classes: int,
        class_weights: Optional[np.array] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        self._class_weights = class_weights
        self._label_smoothing = label_smoothing
        self._n_classes = n_classes

    def initialize(self, device: torch.device):
        # loss
        # note, reduction with class weights != 0 is different to semantic ce
        if self._class_weights is not None:
            self._class_weights = torch.tensor(self._class_weights,
                                               device=device).float()
        self._loss = CrossEntropyLoss(
            weight=self._class_weights,
            label_smoothing=self._label_smoothing,
            ignore_index=-1,    # we ignore void
            reduction='mean'    # average over non-ignored targets
        )
        # metrics (keep it on cpu, it is faster)
        self._metric_cm = ConfusionMatrix(num_classes=self._n_classes)
        # TODO: remove as the bug is fixed in later versions
        # bugfix: confmat is initialized as type float32 (in v0.6.1) which
        # results in slightly varying ground truth counts over several epochs
        # due to accumulating large numbers as float32
        # see: https://github.com/PyTorchLightning/metrics/pull/715
        self._metric_cm._defaults['confmat'] = \
            self._metric_cm._defaults['confmat'].long()
        self._metric_cm.reset()

    def _compute_losses(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:
        # compute loss
        target_shifted = batch['scene'].long() - 1
        total_loss = self._loss(input=predictions_post['scene_output'],
                                target=target_shifted)

        # create dict of unweighted losses (for advanced multi-task learning)
        loss_dict = {
            self.mark_as_total('scene'): total_loss
        }

        return loss_dict

    @append_profile_to_logs('scene_step_time')
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

    @append_profile_to_logs('scene_step_time')
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
        mask = batch['scene'] != 0    # mask of non-void scenes
        if sum(mask) > 0:    # update cm only if there are non-void scenes
            preds = predictions_post['scene_class_idx'][mask]
            target = batch['scene'][mask] - 1    # first apply mask -> -1 is safe
            self._metric_cm.update(preds=preds.cpu(), target=target.cpu().long())

        return loss_dict, {}

    @append_profile_to_logs('scene_epoch_end_time')
    def validation_epoch_end(self):
        # compute acc and bacc
        cm = self._metric_cm.confmat
        tp = torch.diag(cm)
        gt = torch.sum(cm, dim=1)
        # ignore empty classes
        tp = tp[gt != 0]
        gt = gt[gt != 0]
        acc = tp.sum().float() / gt.sum().float()
        bacc = torch.mean(tp.float() / gt.float())

        # create result dicts
        artifacts = {'scene_cm': cm.clone()}
        logs = {'scene_acc': acc, 'scene_bacc': bacc}

        # reset metric (it is not done automatically)
        self._metric_cm.reset()
        assert self._metric_cm.confmat.sum() == 0

        return artifacts, {}, logs
