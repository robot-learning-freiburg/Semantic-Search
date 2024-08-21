# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional

import torch
from torch import nn
from torch import optim


def save_ckpt(filepath: str,
              model: nn.Module,
              optimizer: Optional[optim.Optimizer] = None,
              epoch: Optional[int] = None):
    ckpt = {'state_dict': model.state_dict()}

    if optimizer is not None:
        ckpt['optimizer'] = optimizer.state_dict()
    if epoch is not None:
        ckpt['epoch'] = epoch

    torch.save(ckpt, filepath)
