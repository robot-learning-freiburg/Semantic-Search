# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple

import torch

from .base import LossBase


class L1Loss(LossBase):
    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()
        # reduce later, so we can get l1 per element
        self._loss = torch.nn.L1Loss(reduction='none')
        self._reduction = reduction
        assert self._reduction in ('sum', 'mean', 'none')

    def _compute_loss(
        self,
        input_: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        loss = self._loss(input_, target)

        if 'sum' == self._reduction:
            # first average over channel/feature axis to get the mse per element
            if loss.ndim in (2, 4):
                # bchw/bn: we have a channel/feature axis at index 1
                loss = torch.mean(loss, dim=1)
            n_elements = loss.numel()    # number of pixels
            loss = loss.sum()
        elif 'mean' == self._reduction:
            loss = loss.mean()
            n_elements = 1
        else:
            n_elements = input_.numel()

        return loss, n_elements
