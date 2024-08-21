# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
import random
from typing import Any, Iterator, List, Sized, Tuple, Type

import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data._utils.collate import default_collate


class CollateIgnoredDict(dict):
    """Enables a custom pytorch collate function ignore this dict."""
    pass


def collate(
    data: List[Any],
    type_blacklist: Tuple[Type] = (np.ndarray, CollateIgnoredDict)
) -> Any:
    # input for first call is List[BatchType]

    # get first element for type and dict keys later
    elem = data[0]

    if isinstance(elem, type_blacklist):
        # do not modify blacklisted types, e.g., numpy arrays, keep as list
        return data

    if isinstance(elem, torch.Tensor):
        # list of tensor, check shape
        if not all(t.shape == elem.shape for t in data):
            # we cannot stack the tensors as batch, keep a list of tensors
            return data

    if not isinstance(elem, dict):
        # use default pytorch collate
        return default_collate(data)

    # it is a dict
    return {
        key: collate([s[key] for s in data], type_blacklist)
        for key in elem
    }


class RandomSamplerSubset(Sampler[int]):
    def __init__(
        self,
        data_source: Sized,
        subset: float = 1,
        deterministic: bool = False
    ) -> None:
        self.data_source = data_source
        self.subset = subset
        self.deterministic = deterministic

    def __iter__(self) -> Iterator[int]:
        if self.deterministic:
            # always the same indices on each call but in random order
            seed = 0
        else:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        indices = torch.randperm(len(self.data_source), generator=generator)
        # apply subset
        indices = indices.tolist()[:len(self)]
        # shuffle the indices
        random.shuffle(indices)
        yield from indices

    def __len__(self) -> int:
        return int(len(self.data_source) * self.subset)
