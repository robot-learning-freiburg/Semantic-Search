# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional, Tuple

import torch

from .base import LossBase


class CrossEntropyLossSemantic(LossBase):
    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        weighted_reduction: bool = False    # was used in first ESANet
    ) -> None:
        super().__init__()

        self._weights = weights
        self._weighted_reduction = weighted_reduction

        if weighted_reduction:
            assert self._weights is not None
            self._n_classes = len(self._weights) + 1    # +1 for void
            self._dtype = torch.uint8 if self._n_classes < 2**8 else torch.int16
            reduction = 'none'
        else:
            reduction = 'sum'

        self._loss = torch.nn.CrossEntropyLoss(
            weight=self._weights,
            reduction=reduction,
            ignore_index=-1,
            label_smoothing=label_smoothing
        )

    def _compute_loss(
        self,
        input_: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        # compute loss
        target_shifted = target.long() - 1     # network does not predict void
        loss = self._loss(input_, target_shifted)

        # count number of non-void elements
        n_elements = torch.sum(target_shifted >= 0).cpu().detach().item()

        if not self._weighted_reduction:
            # reduction already done by CrossEntropyLoss -> everything done
            return loss, n_elements

        # reduce using weighted sum
        n_pixels_per_class = torch.bincount(
                target.flatten().type(self._dtype),
                minlength=self._n_classes
        )
        divisor_weighted_pixel_sum = torch.sum(
            n_pixels_per_class[1:] * self._weights    # without void
        )

        # note that 0.0 is assigned to (void) pixels labeled as `ignore_index`
        # in CrossEntropyLoss, moreover, divisor_weighted_pixel_sum ignores void
        # as well, thus the following line is fine
        return loss.sum() / divisor_weighted_pixel_sum, n_elements
