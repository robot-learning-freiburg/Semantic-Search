# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Dict, Union

from copy import deepcopy

import torch

from .base import LossWeightingBase


class FixedLossWeighting(LossWeightingBase):
    def __init__(self, weights: Dict[str, Union[float, torch.Tensor]]) -> None:
        super().__init__()

        self._initial_weights = weights
        self._weights = deepcopy(self._initial_weights)

    @property
    def weights(self) -> Dict[str, Union[float, torch.Tensor]]:
        return self._weights

    def reset_weights(self):
        self._weights = deepcopy(self._initial_weights)

    def reduce_losses(
        self,
        losses: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        weighted_losses = torch.stack(
            [self._weights[key] * losses[key] for key in self._weights]
        )

        return weighted_losses.sum()
