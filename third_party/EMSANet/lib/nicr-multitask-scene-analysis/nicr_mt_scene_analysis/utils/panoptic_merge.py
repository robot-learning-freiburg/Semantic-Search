# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Some parts of this code are based on:
    https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
"""
from typing import Dict, Sequence, List, Tuple

from collections import Counter

import torch
import numpy as np
from scipy import stats


def deeplab_merge_batch(
    semantic_batch: torch.Tensor,
    instance_batch: torch.Tensor,
    instance_fg_batch: torch.Tensor,
    max_instances_per_category: int,
    thing_ids: Sequence[int],
    void_label: int
) -> Tuple[torch.Tensor, List[Dict]]:
    panoptic_preds = []
    panoptic_id_dicts = []
    for semantic_batch_i, instance_batch_i, instance_batch_fg \
            in zip(semantic_batch, instance_batch, instance_fg_batch):
        panoptic_pred, id_dict = \
            deeplab_merge_semantic_and_instance(semantic_batch_i,
                                                instance_batch_i,
                                                instance_batch_fg,
                                                max_instances_per_category,
                                                thing_ids,
                                                void_label)
        panoptic_preds.append(panoptic_pred)
        panoptic_id_dicts.append(id_dict)
    panoptic_preds = torch.stack(panoptic_preds)
    return panoptic_preds, panoptic_id_dicts


def naive_merge_semantic_and_instance_np(
    sem_seg: np.ndarray,
    ins_seg: np.ndarray,
    max_instances_per_category: int,
    thing_ids: Sequence[int],
    void_label: int
) -> Tuple[torch.Tensor, Dict[int, int]]:
    # In case thing mask does not align with semantic prediction.
    pan_seg = np.zeros_like(sem_seg, dtype=np.int64) + void_label

    # Keep track of instance id for each class.
    class_id_tracker = Counter()

    # This dict keeps track of which panoptic id corresponds to which
    # instance id.
    id_dict = {}

    # Paste thing by majority voting.
    instance_ids = np.unique(ins_seg)

    for ins_id in instance_ids:
        if ins_id == 0:
            continue

        thing_mask = (ins_seg == ins_id)
        if len(thing_mask.nonzero()[0]) == 0:
            continue

        semantic_labels = np.unique(sem_seg[thing_mask])
        # Naive approach is to always take the full semantic mask.
        # If a instance label includes more than one semantic label, the
        # instance is divided in multiple parts.
        for class_id in semantic_labels:
            # ignore void
            if class_id == 0:
                continue
            class_id = class_id.astype(np.int64)
            class_id_tracker[class_id.item()] += 1
            new_ins_id = class_id_tracker[class_id.item()]
            panoptic_id = (class_id * max_instances_per_category + new_ins_id)
            id_dict[int(panoptic_id)] = int(ins_id)

            label_mask = (sem_seg == class_id)
            mask = label_mask & thing_mask
            pan_seg[mask] = panoptic_id

    # Paste stuff to unoccupied area.
    class_ids = np.unique(sem_seg)
    for class_id in class_ids:
        # ignore void
        if class_id == 0:
            continue
        if class_id.item() in thing_ids:
            # thing class
            continue
        class_id = class_id.astype(np.int64)
        stuff_mask = (sem_seg == class_id) & (ins_seg == 0)
        pan_seg[stuff_mask] = (class_id * max_instances_per_category)

    return pan_seg, id_dict


def deeplab_merge_semantic_and_instance_np(
    sem_seg: np.ndarray,
    ins_seg: np.ndarray,
    semantic_thing_seg: np.ndarray,
    max_instances_per_category: int,
    thing_ids: List[int],
    void_label: int
) -> Tuple[np.ndarray, Dict[int, int]]:
    # In case thing mask does not align with semantic prediction.
    pan_seg = np.zeros_like(sem_seg, dtype=np.int64) + void_label
    is_thing = (ins_seg > 0) & (semantic_thing_seg > 0)

    # Keep track of instance id for each class.
    class_id_tracker = Counter()

    # This dict keeps track of which panoptic id corresponds to which
    # instance id.
    id_dict = {}

    # Paste thing by majority voting.
    instance_ids = np.unique(ins_seg)

    for ins_id in instance_ids:
        if ins_id == 0:
            continue
        # Make sure only do majority voting within `semantic_thing_seg`.
        thing_mask = (ins_seg == ins_id) & is_thing
        if len(np.nonzero(thing_mask)[0]) == 0:
            continue
        class_id = stats.mode(sem_seg[thing_mask].flatten()).mode
        # ignore void
        if class_id == 0:
            continue
        class_id = class_id.astype(np.int64)
        class_id_tracker[class_id.item()] += 1
        new_ins_id = class_id_tracker[class_id.item()]
        panoptic_id = (class_id * max_instances_per_category + new_ins_id)
        id_dict[int(panoptic_id)] = int(ins_id)
        pan_seg[thing_mask] = panoptic_id

    # Paste stuff to unoccupied area.
    class_ids = np.unique(sem_seg)
    for class_id in class_ids:
        # ignore void
        if class_id == 0:
            continue
        if class_id.item() in thing_ids:
            # thing class
            continue
        class_id = class_id.astype(np.int64)
        stuff_mask = (sem_seg == class_id) & (ins_seg == 0)
        pan_seg[stuff_mask] = (class_id * max_instances_per_category)

    return pan_seg, id_dict


def deeplab_merge_semantic_and_instance(
    sem_seg: torch.Tensor,
    ins_seg: torch.Tensor,
    semantic_thing_seg: torch.Tensor,
    max_instances_per_category: int,
    thing_ids: Sequence[int],
    void_label: int
) -> Tuple[torch.Tensor, Dict[int, int]]:
    # In case thing mask does not align with semantic prediction.
    pan_seg = torch.zeros_like(sem_seg, dtype=torch.long) + void_label
    is_thing = (ins_seg > 0) & (semantic_thing_seg > 0)

    # Keep track of instance id for each class.
    class_id_tracker = Counter()

    # This dict keeps track of which panoptic id corresponds to which
    # instance id.
    id_dict = {}

    # Paste thing by majority voting.
    instance_ids = torch.unique(ins_seg)

    for ins_id in instance_ids:
        if ins_id == 0:
            continue
        # Make sure only do majority voting within `semantic_thing_seg`.
        thing_mask = (ins_seg == ins_id) & is_thing
        if torch.nonzero(thing_mask).size(0) == 0:
            continue
        class_id, _ = torch.mode(sem_seg[thing_mask].view(-1))
        # ignore void
        if class_id == 0:
            continue
        class_id = class_id.long()
        class_id_tracker[class_id.item()] += 1
        new_ins_id = class_id_tracker[class_id.item()]
        panoptic_id = (class_id * max_instances_per_category + new_ins_id)
        id_dict[int(panoptic_id)] = int(ins_id)
        pan_seg[thing_mask] = panoptic_id

    # Paste stuff to unoccupied area.
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        # ignore void
        if class_id == 0:
            continue
        if class_id.item() in thing_ids:
            # thing class
            continue
        class_id = class_id.long()
        stuff_mask = (sem_seg == class_id) & (ins_seg == 0)
        pan_seg[stuff_mask] = (class_id * max_instances_per_category)

    return pan_seg, id_dict
