# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from math import ceil
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from emsanet.lovasz_loses import lovasz_softmax
from nicr_mt_scene_analysis.data.preprocessing.resize import (
    get_fullres,
    get_fullres_key,
)
from nicr_mt_scene_analysis.loss.base import LossBase
from nicr_mt_scene_analysis.metric import MeanIntersectionOverUnion
from nicr_mt_scene_analysis.task_helper.base import (
    TaskHelperBase,
    append_detached_losses_to_logs,
    append_profile_to_logs,
)
from nicr_mt_scene_analysis.types import BatchType
from nicr_mt_scene_analysis.visualization import (
    visualize_heatmap_pil,
    visualize_semantic_pil,
)


class CrossEntropyLossSemantic(LossBase):
    def __init__(
        self,
        iters_per_epoch: int,
        saturate_unc_loss_epoch: int,
        weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        ohem: float = 0.0,
        lovasz: bool = False,
        weighted_ohem: bool = True,
        uncertainity_loss: bool = False,
        uncertainity_coef: float = 0.03,
    ) -> None:
        super().__init__()

        if ohem > 0.0 and not weighted_ohem:
            weights = None
            print(
                "Forcing semantic class weights to None because OHEM is active and we're not using weighted OHEM."
            )

        self._weights = weights
        self._ohem = ohem
        self._lovasz = lovasz
        self._cross_entropy = torch.nn.CrossEntropyLoss(
            weight=self._weights,
            reduction="none",
            ignore_index=-1,
            label_smoothing=label_smoothing,
        )
        self._uncertainity_loss = uncertainity_loss
        self.iters_per_epoch = iters_per_epoch
        self.saturate_unc_loss_epoch = saturate_unc_loss_epoch
        self.uncertainity_coef = uncertainity_coef

    def _compute_loss(
        self,
        input_: torch.Tensor,
        target: torch.Tensor,
        batch_idx: int,
        epoch_idx: int,
    ) -> Tuple[torch.Tensor, int]:
        # compute loss
        target_shifted = target.long() - 1  # network does not predict void
        # count number of non-void elements
        n_elements = torch.sum(target_shifted >= 0).cpu().detach().item()
        if self._uncertainity_loss:
            device = input_.device
            gt_onehot, _ = self._expand_onehot_labels(target_shifted, input_.shape, -1)
            gt_onehot = gt_onehot.to(device)
            evidence = F.softplus(input_)
            num_classes = evidence.shape[1]
            alpha = evidence + 1
            pixel_level_loss = self.edl_loss(
                torch.log, gt_onehot, alpha, num_classes, batch_idx, epoch_idx, device
            )
        else:
            pixel_level_loss = self._cross_entropy(input_, target_shifted)

        if self._ohem > 0.0:
            pixel_loss = pixel_level_loss.view(-1)
            top_k = int(ceil(pixel_loss.numel() * self._ohem))
            top_k_losses, top_k_indices = torch.topk(pixel_loss, top_k)
            n_elements = (
                torch.sum(target_shifted.view(-1)[top_k_indices] >= 0)
                .cpu()
                .detach()
                .item()
            )
            loss = top_k_losses.sum()
        else:
            loss = pixel_level_loss.sum()

        if self._lovasz:
            lovaz_loss = (
                lovasz_softmax(torch.softmax(input_, dim=-1), target_shifted, ignore=-1)
                * n_elements
            )  # will be divided by n_elements later

            loss = (loss + lovaz_loss) * 0.5
        return loss, n_elements

    def _expand_onehot_labels(self, labels, target_shape, ignore_index):
        """Expand onehot labels to match the size of prediction."""
        bin_labels = labels.new_zeros(target_shape)
        valid_mask = (labels >= 0) & (labels != ignore_index)
        inds = torch.nonzero(valid_mask, as_tuple=True)
        labels = labels.long()
        if inds[0].numel() > 0:
            if labels.dim() == 3:
                bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
            else:
                bin_labels[inds[0], labels[valid_mask]] = 1

        valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
        bin_label_weights = valid_mask
        return bin_labels, bin_label_weights

    def edl_loss(self, func, y, alpha, num_classes, curr_iter, curr_epoch, device=None):
        y = y.to(device)
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
        it_per_epoch = self.iters_per_epoch
        total_epochs = self.saturate_unc_loss_epoch
        saturate_total_batches = total_epochs * it_per_epoch
        current_batch = ((curr_epoch) * it_per_epoch) + curr_iter
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(current_batch / saturate_total_batches, dtype=torch.float32),
        ).to(device)

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = (
            (torch.tensor([[self.uncertainity_coef]]).to(device))
            * annealing_coef
            * self.kl_divergence(kl_alpha, num_classes, device=device)
        )
        return A + kl_div

    def kl_divergence(self, alpha, num_classes, device=None):
        if not device:
            device = alpha.device
        ones = torch.ones(alpha.size(), dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl

    def forward(
        self,
        input_tensors: Sequence[torch.Tensor],
        target_tensors: Sequence[torch.Tensor],
        batch_idx: int,
        epoch_idx: int,
    ) -> Tuple[Tuple[torch.Tensor, int]]:
        # determine loss for all scales
        return tuple(
            self._compute_loss(input_, target, batch_idx, epoch_idx)
            for input_, target in zip(input_tensors, target_tensors)
        )


class SemanticTaskHelper(TaskHelperBase):
    def __init__(
        self,
        n_classes: int,
        iters_per_epoch: int,
        saturate_unc_loss_epoch: int = -1,
        uncertainity_loss: bool = False,
        class_weights: Optional[np.array] = None,
        ohem: float = 0.0,
        lovasz: bool = False,
        weighted_ohem: bool = False,
        label_smoothing: float = 0.0,
        uncertainity_coef: float = 0.03,
        disable_multiscale_supervision: bool = False,
        examples_cmap: Union[Sequence[Tuple[int, int, int]], np.array] = None,
    ) -> None:
        super().__init__()

        self._n_classes = n_classes
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing
        self._disable_multiscale_supervision = disable_multiscale_supervision

        # during validation, we store some examples for visualization purposes
        self._examples = {}
        self._examples_cmap = examples_cmap
        self._ohem = ohem
        self._lovasz = lovasz
        self._weighted_ohem = weighted_ohem
        self._iters_per_epoch = iters_per_epoch
        self._saturated_unc_loss_epoch = saturate_unc_loss_epoch
        self._uncertainity_loss = uncertainity_loss
        self._uncertainity_coef = uncertainity_coef

    def initialize(self, device: torch.device):
        # loss
        if self._class_weights is not None:
            self._class_weights = torch.tensor(
                self._class_weights, device=device
            ).float()
        self._loss = CrossEntropyLossSemantic(
            weights=self._class_weights,
            label_smoothing=self._label_smoothing,
            ohem=self._ohem,
            lovasz=self._lovasz,
            weighted_ohem=self._weighted_ohem,
            saturate_unc_loss_epoch=self._saturated_unc_loss_epoch,
            iters_per_epoch=self._iters_per_epoch,
            uncertainity_loss=self._uncertainity_loss,
            uncertainity_coef=self._uncertainity_coef,
        )
        # metrics (keep it on cpu, it is faster)
        self._metric_iou = MeanIntersectionOverUnion(n_classes=self._n_classes)
        # self._metric_iou = self._metric_iou.to(device)
        self._metric_iou.reset()
        self._current_epoch = 0

    def _compute_losses(
        self, batch: BatchType, batch_idx: int, predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:
        # collect predictions and targets for loss
        no_multiscale = self._disable_multiscale_supervision
        preds, targets, keys = self.collect_predictions_and_targets_for_loss(
            batch=batch,
            batch_key="semantic",
            predictions_post=predictions_post,
            predictions_post_key="semantic_output",
            side_outputs_key=None if no_multiscale else "semantic_side_outputs",
        )
        # compute losses
        loss_outputs = self._loss(
            input_tensors=preds,
            target_tensors=targets,
            batch_idx=batch_idx,
            epoch_idx=self.current_epoch_idx,
        )

        # create loss dict
        loss_dict = {
            f"semantic_loss_{key}": loss / n
            for key, (loss, n) in zip(keys, loss_outputs)
        }

        # compute total loss (accumulate losses of all side outputs)
        total_loss = self.accumulate_losses(
            losses=[loss for loss, _ in loss_outputs],
            n_elements=[n for _, n in loss_outputs],
        )

        # append unweighted total loss (for advanced multi-task learning)
        loss_dict[self.mark_as_total("semantic")] = total_loss

        return loss_dict

    @append_profile_to_logs("semantic_step_time")
    @append_detached_losses_to_logs()
    def training_step(
        self, batch: BatchType, batch_idx: int, predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # compute loss
        loss_dict = self._compute_losses(
            batch=batch, batch_idx=batch_idx, predictions_post=predictions_post
        )
        return loss_dict, {}

    @append_profile_to_logs("semantic_step_time")
    @append_detached_losses_to_logs()
    def validation_step(
        self, batch: BatchType, batch_idx: int, predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # compute loss
        loss_dict = self._compute_losses(
            batch=batch, batch_idx=batch_idx, predictions_post=predictions_post
        )

        # update metric
        target = get_fullres(batch, "semantic")
        mask = target != 0  # mask of non-void pixels
        preds = predictions_post[get_fullres_key("semantic_segmentation_idx")][mask]
        target = target[mask] - 1  # first apply mask -> -1 is safe
        self._metric_iou.update(preds=preds.cpu(), target=target.cpu())

        # store example for visualization (not fullres!)
        if batch_idx == 0:
            # class
            ex = predictions_post["semantic_segmentation_idx"][0]
            key = f"semantic_example_batch_idx_{batch_idx}_0"
            self._examples[key] = visualize_semantic_pil(
                semantic_img=ex.cpu().numpy(), colors=self._examples_cmap
            )

            # score
            ex = predictions_post["semantic_segmentation_score"][0]
            key = f"semantic_example_batch_score_{batch_idx}_0"
            self._examples[key] = visualize_heatmap_pil(
                heatmap_img=ex.cpu().numpy(), min_=0, max_=1
            )

        return loss_dict, {}

    @append_profile_to_logs("semantic_epoch_end_time")
    def validation_epoch_end(self):
        miou, ious = self._metric_iou.compute(return_ious=True)
        logs = {"semantic_miou": miou}
        artifacts = {
            "semantic_cm": self._metric_iou.confmat.clone(),
            "semantic_ious_per_class": ious.clone(),
        }

        # reset metric (it is not done automatically)
        self._metric_iou.reset()
        assert self._metric_iou.confmat.sum() == 0

        return artifacts, self._examples, logs
