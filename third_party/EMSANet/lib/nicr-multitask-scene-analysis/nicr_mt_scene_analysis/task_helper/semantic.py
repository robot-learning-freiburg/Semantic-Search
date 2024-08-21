# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..data.preprocessing.resize import get_fullres
from ..data.preprocessing.resize import get_fullres_key
from ..loss import CrossEntropyLossSemantic
from ..metric import MeanIntersectionOverUnion
from ..types import BatchType
from ..visualization import visualize_semantic_pil
from ..visualization import visualize_heatmap_pil
from .base import append_detached_losses_to_logs
from .base import append_profile_to_logs
from .base import TaskHelperBase


class SemanticTaskHelper(TaskHelperBase):
    def __init__(
        self,
        n_classes: int,
        class_weights: Optional[np.array] = None,
        label_smoothing: float = 0.0,
        disable_multiscale_supervision: bool = False,
        examples_cmap: Union[Sequence[Tuple[int, int, int]], np.array] = None
    ) -> None:
        super().__init__()

        self._n_classes = n_classes
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing
        self._disable_multiscale_supervision = disable_multiscale_supervision

        # during validation, we store some examples for visualization purposes
        self._examples = {}
        self._examples_cmap = examples_cmap

    def initialize(self, device: torch.device):
        # loss
        if self._class_weights is not None:
            self._class_weights = torch.tensor(self._class_weights,
                                               device=device).float()
        self._loss = CrossEntropyLossSemantic(
            weights=self._class_weights,
            label_smoothing=self._label_smoothing
        )
        # metrics (keep it on cpu, it is faster)
        self._metric_iou = MeanIntersectionOverUnion(n_classes=self._n_classes)
        # self._metric_iou = self._metric_iou.to(device)
        self._metric_iou.reset()

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
            batch_key='semantic',
            predictions_post=predictions_post,
            predictions_post_key='semantic_output',
            side_outputs_key=None if no_multiscale else 'semantic_side_outputs'
        )
        # compute losses
        loss_outputs = self._loss(input_tensors=preds, target_tensors=targets)

        # create loss dict
        loss_dict = {
            f'semantic_loss_{key}': loss/n
            for key, (loss, n) in zip(keys, loss_outputs)
        }

        # compute total loss (accumulate losses of all side outputs)
        total_loss = self.accumulate_losses(
            losses=[loss for loss, _ in loss_outputs],
            n_elements=[n for _, n in loss_outputs]
        )

        # append unweighted total loss (for advanced multi-task learning)
        loss_dict[self.mark_as_total('semantic')] = total_loss

        return loss_dict

    @append_profile_to_logs('semantic_step_time')
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

    @append_profile_to_logs('semantic_step_time')
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
        target = get_fullres(batch, 'semantic')
        mask = target != 0    # mask of non-void pixels
        preds = predictions_post[get_fullres_key('semantic_segmentation_idx')][mask]
        target = target[mask] - 1    # first apply mask -> -1 is safe
        self._metric_iou.update(preds=preds.cpu(), target=target.cpu())

        # store example for visualization (not fullres!)
        if batch_idx == 0:
            # class
            ex = predictions_post['semantic_segmentation_idx'][0]
            key = f'semantic_example_batch_idx_{batch_idx}_0'
            self._examples[key] = visualize_semantic_pil(
                semantic_img=ex.cpu().numpy(),
                colors=self._examples_cmap
            )

            # score
            ex = predictions_post['semantic_segmentation_score'][0]
            key = f'semantic_example_batch_score_{batch_idx}_0'
            self._examples[key] = visualize_heatmap_pil(
                heatmap_img=ex.cpu().numpy(),
                min_=0, max_=1
            )

        return loss_dict, {}

    @append_profile_to_logs('semantic_epoch_end_time')
    def validation_epoch_end(self):
        miou, ious = self._metric_iou.compute(return_ious=True)
        logs = {'semantic_miou': miou}
        artifacts = {'semantic_cm': self._metric_iou.confmat.clone(),
                     'semantic_ious_per_class': ious.clone()}

        # reset metric (it is not done automatically)
        self._metric_iou.reset()
        assert self._metric_iou.confmat.sum() == 0

        return artifacts, self._examples, logs
