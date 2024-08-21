# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Dict, Sequence, Union

from collections import deque
from copy import deepcopy

import torch

from .base import LossWeightingBase


class DynamicWeightAverage(LossWeightingBase):
    def __init__(
        self,
        loss_keys_to_consider: Sequence[str],
        temperature: float = 2.0
    ) -> None:
        # see: https://arxiv.org/pdf/1803.10704.pdf
        super().__init__()

        self._loss_keys = loss_keys_to_consider
        self._temperature = temperature

        self._loss_history = deque([], maxlen=2)    # for determining weights
        self._loss_buffer = []      # stores the losses within an epoch

        # init weights
        self._default_weights = {k: 1.0 for k in self._loss_keys}
        self._weights = deepcopy(self._default_weights)

    @property
    def weights(self) -> Dict[str, Union[float, torch.Tensor]]:
        return self._weights

    def reset_weights(self):
        self._loss_buffer = []
        self._loss_history = deque([], maxlen=2)
        self._weights = deepcopy(self._default_weights)

    def _compute_weights(self) -> None:
        # update loss history (take mean of epoch)
        if self._loss_buffer:
            self._loss_history.append({
                k: torch.mean(
                    torch.stack([losses[k] for losses in self._loss_buffer])
                )
                for k in self._loss_keys
            })
            # print(self._loss_history)

        # compute weights
        if len(self._loss_history) < 2:
            # not enough values in loss history, assign default weights
            self._weights = deepcopy(self._default_weights)
            return

        # dwa
        # note only as of python >= 3.7: dicts are always ordered !
        weights = torch.stack([
            # t-1 / t-2
            self._loss_history[-1][k] / self._loss_history[-2][k]
            for k in self._loss_keys
        ])
        weights = len(weights)*torch.nn.functional.softmax(
            weights/self._temperature, dim=-1
        )
        self._weights = {k: w.item() for k, w in zip(self._loss_keys,
                                                     weights)}

    def reduce_losses(
        self,
        losses: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        if 0 == batch_idx:
            # compute new weights at the beginning of a new epoch
            # note: a sanity check breaks this smart hack compute new weights,
            # use `reset_weights` to reset after performing the sanity check
            self._compute_weights()

        # store current losses within the same epoch in loss buffer
        detached_losses = {k: losses[k].detach().clone()
                           for k in self._loss_keys}
        if len(self._loss_buffer) == batch_idx:
            # we still must enlarge the buffer
            self._loss_buffer.append(detached_losses)
        else:
            # replace element in buffer
            # note, we assume same length for all epochs
            self._loss_buffer[batch_idx] = detached_losses

        # reduce loss
        total_loss = torch.sum(
            torch.stack([self.weights[key] * losses[key]
                         for key in self._loss_keys])
        )

        return total_loss
