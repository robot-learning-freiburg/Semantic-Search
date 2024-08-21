# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch


def unit_length(x: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))

    return x / (norm + epsilon)
