# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import abc
from typing import List, Union

from torch import Tensor
import torch.nn as nn


class Backbone(abc.ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @property
    @abc.abstractmethod
    def stages(self) -> List[Union[nn.Sequential, nn.Module]]:
        pass

    @property
    @abc.abstractmethod
    def stages_n_channels(self) -> List[int]:
        pass

    @property
    @abc.abstractmethod
    def stages_downsampling(self) -> List[int]:
        pass

    @property
    @abc.abstractmethod
    def stages_memory_layout(self) -> List[str]:
        pass

    def forward_stage(self, stage_idx: int, x: Tensor) -> Tensor:
        stage = self.stages[stage_idx]
        return stage(x)

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.stages)):
            x = self.forward_stage(i, x)
        return x
