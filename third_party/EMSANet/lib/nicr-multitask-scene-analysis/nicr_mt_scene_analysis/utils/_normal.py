# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch

from ._torch import unit_length


class NormalOutputNormalization(torch.nn.Module):
    def __init__(self, epsilon=1e-7) -> None:
        self._epsilon = epsilon
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return unit_length(x, self._epsilon)
