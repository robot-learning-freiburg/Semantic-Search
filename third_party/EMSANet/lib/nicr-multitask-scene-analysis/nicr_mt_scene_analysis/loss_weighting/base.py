# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Dict, Union

import abc

import torch


class LossWeightingBase(abc.ABC, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, device: torch.device):
        # move additional things to correct device
        # note, currently not used as the loss reduction is done on cpu
        pass

    @property
    @abc.abstractmethod
    def weights(self) -> Union[Dict[str, Union[float, torch.Tensor]], None]:
        pass

    @abc.abstractmethod
    def reset_weights(self):
        pass

    @abc.abstractmethod
    def reduce_losses(
        self,
        losses: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        pass

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        return self.reduce_losses(losses, batch_idx)
