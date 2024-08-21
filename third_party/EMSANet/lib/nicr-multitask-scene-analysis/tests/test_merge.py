# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import torch
import numpy as np
import pytest
from tqdm import tqdm

from nicr_mt_scene_analysis.data.preprocessing import PanopticTargetGenerator
from nicr_mt_scene_analysis.utils import panoptic_merge

from test_metrics import get_dataloader_for_tasks


def batch_to_sample(batch, idx):
    r_dict = {}
    for key, value in batch.items():
        el = value[idx]
        if isinstance(el, torch.Tensor):
            el = el.cpu().numpy()
        r_dict[key] = el
    return r_dict


@pytest.mark.parametrize('name', ('nyuv2', 'sunrgbd'))
def test_compare_merge_methodes(name):
    dataloader = get_dataloader_for_tasks(('semantic', 'instance'), name=name)
    dataset_config = dataloader.dataset.config
    n_semantic_classes = len(dataset_config.semantic_label_list)
    is_thing = dataset_config.semantic_label_list.classes_is_thing
    max_instances_per_category = (1 << 16)
    thing_ids = np.where(is_thing)[0]
    void_label = 0
    pan_gen = PanopticTargetGenerator(is_thing)

    for batch in tqdm(dataloader):
        semantic_batch = batch['semantic']
        instance_batch = batch['instance']
        for idx, (semantic, instance) in enumerate(zip(semantic_batch, instance_batch)):
            instance_fg = instance != 0

            # use naive merge
            naive_np, naive_dict_np = \
                panoptic_merge.naive_merge_semantic_and_instance_np(
                    semantic.numpy(),
                    instance.numpy(),
                    max_instances_per_category,
                    thing_ids,
                    void_label
                )

            # use panoptic deeplab numpy merge
            deeplab_np, deeplab_dict_np = \
                panoptic_merge.deeplab_merge_semantic_and_instance_np(
                    semantic.numpy(),
                    instance.numpy(),
                    instance_fg.numpy(),
                    max_instances_per_category,
                    thing_ids,
                    void_label
                )

            # use panoptic deeplab torch merge
            deeplab_torch, deeplab_dict_torch = \
                panoptic_merge.deeplab_merge_semantic_and_instance(
                    semantic,
                    instance,
                    instance_fg,
                    max_instances_per_category,
                    thing_ids,
                    void_label
                )

            sample_target = pan_gen(batch_to_sample(batch, idx))
            sample_panoptic = sample_target['panoptic']
            sample_panoptic_ids = sample_target['panoptic_ids_to_instance_dict']

            assert (naive_np == deeplab_np).all()
            assert (naive_np == deeplab_torch.numpy()).all()
            assert (sample_panoptic == naive_np).all()
            assert naive_dict_np == deeplab_dict_np
            assert naive_dict_np == deeplab_dict_torch
            assert naive_dict_np == sample_panoptic_ids
