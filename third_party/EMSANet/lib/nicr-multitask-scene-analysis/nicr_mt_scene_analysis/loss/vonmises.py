# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Parts of this code are based on:
    Beyer et al.: Biternion Nets: Continuous Head Pose Regression from
    Discrete Training Labels, GCPR 2015.
    -> http://www.spencer.eu/papers/beyerBiternions15.pdf
"""
from typing import Tuple

import torch

from .base import LossBase


class VonMisesLossBiternion(LossBase):
    def __init__(self, reduction: str = 'sum', kappa: float = 1.0) -> None:
        super().__init__()

        assert reduction in ('sum', 'none')

        self._kappa = kappa
        self._reduction = reduction

    def _compute_loss(
        self,
        input_: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        if input_.ndim != 2 or target.ndim != 2:
            # force 2d inputs with shape (b*h*w, 2) as this enables masking
            # values before quite easily
            raise ValueError(
                "VonMisesLossBiternion does only support 2d inputs with shape "
                "(n, 2), you can transpose your input to channels last and "
                "reshape to shape (b*h*w, c=2), e.g., (b, c, h, w) -> "
                "(b, h, w, c) -> (b*h*w, c) with c = 2."
            )

        # for normalized inputs and targets between -1 (worst) and 1 (best)
        cos_angles = (input_*target).sum(dim=1, keepdim=True)
        cos_angles = torch.exp(self._kappa * (cos_angles - 1))
        score = 1 - cos_angles

        n_elements = score.numel()
        if 'sum' == self._reduction:
            score = score.sum()

        return score, n_elements
