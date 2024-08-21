# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Dict, Sequence, Union

import torch

from .base import LossWeightingBase


class RandomLossWeighting(LossWeightingBase):
    def __init__(
        self,
        loss_keys_to_consider: Sequence[str],
        temperature: float = 1.0,
        scale: bool = False
    ) -> None:
        # see: https://arxiv.org/abs/2111.10603
        # sounds stupid but at least seems not to break training
        # however, in our tests no improvement compared to fixed weights
        super().__init__()

        self._loss_keys = loss_keys_to_consider
        self._temperature = temperature     # 1 in paper
        self._scale = scale

        # init weights
        self._weights = None
        self.reset_weights()

    @property
    def weights(self) -> Dict[str, Union[float, torch.Tensor]]:
        return self._weights

    def reset_weights(self):
        # assign new random weights
        self._compute_weights()

    def _compute_weights(self) -> None:
        weights = torch.nn.functional.softmax(
            torch.randn(len(self._loss_keys))/self._temperature, dim=-1
        )
        if self._scale:
            weights *= len(weights)    # not done in paper

        self._weights = {k: w.item() for k, w in zip(self._loss_keys,
                                                     weights)}

    def reduce_losses(
        self,
        losses: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        # dermine new weights
        self._compute_weights()

        # reduce loss
        total_loss = torch.sum(
            torch.stack([self.weights[key] * losses[key]
                         for key in self._loss_keys])
        )

        return total_loss
