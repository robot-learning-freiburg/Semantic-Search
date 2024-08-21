# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Iterable, Optional
from copy import deepcopy

from ...types import BatchType


DEFAULT_CLONE_KEY = '_no_preprocessing'


def clone_entries(
    sample: BatchType,
    keys_to_clone: Iterable[str],
    ignore_missing_keys: bool = False
) -> BatchType:
    return {
        key: deepcopy(sample[key])
        for key in keys_to_clone
        if key in sample or not ignore_missing_keys
    }


class CloneEntries:
    def __init__(
        self,
        keys_to_clone: Optional[Iterable[str]] = None,
        ignore_missing_keys: bool = False,
        clone_key: str = DEFAULT_CLONE_KEY
    ) -> None:
        self._keys_to_clone = keys_to_clone
        self._ignore_missing_keys = ignore_missing_keys
        self._clone_key = clone_key

    @property
    def clone_key(self):
        return self._clone_key

    def __call__(self, sample: BatchType) -> BatchType:
        # copy all entries such that they are available for later
        keys = self._keys_to_clone or tuple(sample.keys())
        sample[self._clone_key] = clone_entries(sample, keys,
                                                self._ignore_missing_keys)

        return sample


class FlatCloneEntries:
    def __init__(
        self,
        keys_to_clone: Optional[Iterable[str]] = None,
        ignore_missing_keys: bool = False,
        key_prefix: Optional[str] = None,
        key_suffix: Optional[str] = None
    ) -> None:
        assert key_prefix or key_suffix

        self._keys_to_clone = keys_to_clone
        self._ignore_missing_keys = ignore_missing_keys
        self._key_prefix = key_prefix or ''
        self._key_suffix = key_suffix or ''

    def __call__(self, sample: BatchType) -> BatchType:
        # clone entries
        keys = self._keys_to_clone or tuple(sample.keys())
        cloned_entries = clone_entries(sample, keys,
                                       self._ignore_missing_keys)

        # add entries with modified key to dict
        sample.update({
            f'{self._key_prefix}{k}{self._key_suffix}': v
            for k, v in cloned_entries.items()
        })

        return sample
