# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import abc
from typing import Sequence, Tuple

import torch


class LossBase(abc.ABC, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def _compute_loss(
        self,
        input_: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        # tuple of tuple(loss, #loss_elements)
        pass

    def forward(
        self,
        input_tensors: Sequence[torch.Tensor],
        target_tensors: Sequence[torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor, int]]:
        # determine loss for all scales
        return tuple(
            self._compute_loss(input_, target)
            for input_, target in zip(input_tensors, target_tensors)
        )
