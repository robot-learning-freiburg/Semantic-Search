# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional, Sequence

import torch

from ..types import BatchType


def infer_batch_size(batch: BatchType, key: Optional[str] = None) -> int:
    if key is not None:
        # use given key for infering the batch size
        return batch[key].shape[0]

    # default case: we always have 'rgb' and/or 'depth' as batched tensor
    # in `batch`
    tensor = batch.get('rgb', batch.get('depth', None))
    return tensor.shape[0]


def move_batch_to_device(
    batch: BatchType,
    device: torch.device,
    keys_to_ignore: Optional[Sequence[str]] = None,
    keys_to_ignore_recursive: Optional[bool] = True
) -> BatchType:
    assert isinstance(batch, dict), "Not implemented"

    for key in list(batch.keys()):
        if keys_to_ignore is not None and key in keys_to_ignore:
            continue

        if isinstance(batch[key], dict):
            # nested dict
            to_ignore = keys_to_ignore if keys_to_ignore_recursive else None
            batch[key] = move_batch_to_device(
                batch=batch[key],
                device=device,
                keys_to_ignore=to_ignore
            )

        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    return batch
