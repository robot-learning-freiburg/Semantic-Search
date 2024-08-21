# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Tuple

from functools import partial

import nicr_scene_analysis_datasets.pytorch as datasets
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT
from nicr_scene_analysis_datasets.dataset_base import OrientationDict
import numpy as np
from torch.utils.data import DataLoader

from ..data import mt_collate


def get_dataset(name: str,
                split: str,
                sample_keys: Tuple[str],
                **kwargs: Any):
    assert name in datasets.KNOWN_DATASETS

    dataset_kwargs_default = {
        'sunrgbd': {},
        'nyuv2': {'semantic_n_classes': 40},
        'cityscapes': {
            'n_classes': 19,
            'disparity_instead_of_depth': False
        },
        'scenenetrgbd': {'semantic_n_classes': 13},
        'hypersim': {'subsample': 5 if split == 'train' else None},
        'coco': {},
    }

    Dataset = datasets.get_dataset_class(name)
    dataset_kwargs = dataset_kwargs_default[name]
    dataset_kwargs.update(kwargs)
    dataset_path = DATASET_PATH_DICT[name]

    return Dataset(dataset_path=dataset_path,
                   split=split,
                   sample_keys=sample_keys,
                   **dataset_kwargs)


def get_dataloader(dataset, **kwargs):
    collate_fn = partial(mt_collate,
                         type_blacklist=(np.ndarray, OrientationDict))
    dataloader_kwargs = {
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 4,
        'persistent_workers': True,
        'drop_last': False,
        'collate_fn': collate_fn
    }
    dataloader_kwargs.update(kwargs)

    return DataLoader(dataset, **dataloader_kwargs)
