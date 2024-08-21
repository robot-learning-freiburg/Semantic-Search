# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import matplotlib.pyplot as plt
import pytest
import torch

from nicr_mt_scene_analysis.model.postprocessing.instance import InstancePostprocessing
from nicr_mt_scene_analysis.testing import SHOW_RESULTS


def dim_sort(input_center_list):
    # Sort every axis for every tensor in list
    output_tensor_list = []
    for el in input_center_list:
        for axis in reversed(range(len(el.shape))):
            el = el.sort(dim=axis)[0]
        output_tensor_list.append(el)
    return output_tensor_list


def generate_random_batch(batch, height, width, num_inst_min, num_inst_max,
                          num_inst_size_min, num_inst_size_max):
    # Generate random rectangular instances, offsets and heatmaps
    shape_one = (batch, 1, height, width)
    shape_two = (batch, 2, height, width)
    instances_batch = torch.zeros(shape_one, dtype=torch.uint8)
    heatmap_batch = torch.zeros(shape_one)
    offset_batch = torch.zeros(shape_two)
    forgeground_mask_batch = torch.zeros(shape_one, dtype=torch.bool)

    y_coord = torch.ones_like(instances_batch[0, 0], dtype=torch.float32)
    x_coord = torch.ones_like(instances_batch[0, 0], dtype=torch.float32)
    y_coord = torch.cumsum(y_coord, axis=0) - 1
    x_coord = torch.cumsum(x_coord, axis=1) - 1
    centers = []
    for n_batch in range(batch):
        # num_el = torch.randint(low=num_inst_min, high=num_inst_max, size=(1,))
        num_el = 2
        center_x = torch.randint(low=0, high=width, size=(num_el,))
        center_y = torch.randint(low=0, high=height, size=(num_el,))
        instance_size = torch.randint(low=num_inst_size_min,
                                      high=num_inst_size_max,
                                      size=(num_el,))

        centers_tmp = []
        for inst_id, (y, x, size) in enumerate(zip(center_y,
                                                   center_x,
                                                   instance_size)):

            min_x = int(max(x - size, 0))
            max_x = int(min(x + size, width))
            min_y = int(max(y - size, 0))
            max_y = int(min(y + size, height))

            x = int(max_x-(max_x - min_x)/2)
            y = int(max_y-(max_y - min_y)/2)
            heatmap_batch[n_batch, 0, y, x] = 1
            centers_tmp.append((y, x))

            instances_batch[n_batch, 0, min_y:max_y, min_x:max_x] = inst_id + 1
            offset_batch[n_batch, 0, min_y:max_y, min_x:max_x] = y - y_coord[min_y:max_y, min_x:max_x]
            offset_batch[n_batch, 1, min_y:max_y, min_x:max_x] = x - x_coord[min_y:max_y, min_x:max_x]
            forgeground_mask_batch[n_batch, 0, min_y:max_y, min_x:max_x] = True
        centers.append(torch.tensor(centers_tmp))
        assert len(centers_tmp) == num_el

    return (instances_batch, heatmap_batch, offset_batch,
            forgeground_mask_batch, centers)


def show_results(img=None, img_ref=None):
    if not SHOW_RESULTS:
        return

    _, axes = plt.subplots(1, 2)
    if img is not None:
        axes[0].imshow(img, interpolation='nearest')
        axes[0].set_title('Output', fontsize=8)
    if img_ref is not None:
        axes[1].imshow(img_ref, interpolation='nearest')
        axes[1].set_title('Reference', fontsize=8)
    plt.show()


@pytest.mark.parametrize('batch_size', [1, 8])
@pytest.mark.parametrize('num_inst', [(1, 10), (10, 20), (20, 64)])
@pytest.mark.parametrize('num_inst_size', [(5, 15), (20, 40), (30, 80)])
def test_instance_postprocessing(batch_size, num_inst, num_inst_size):

    num_inst_min, num_inst_max = num_inst
    num_inst_size_min, num_inst_size_max = num_inst_size
    h = 480
    w = 640
    post = InstancePostprocessing(normalized_offset=False)

    r_val = generate_random_batch(batch_size, h, w, num_inst_min, num_inst_max,
                                  num_inst_size_min, num_inst_size_max)

    instances, heatmap, offset, forgeground_mask, centers = r_val

    # check instance centers first
    _, found_centers = post._get_instance_centers(heatmap)
    # check that all points are found correctly
    found_centers_sorted = dim_sort(found_centers)
    centers = dim_sort(centers)
    for fc, c in zip(found_centers_sorted, centers):
        assert torch.all(fc == c)

    # check entire instance segmentation
    batch = {
        'instance_foreground': forgeground_mask,
        # we need at least one fullres key in the batch for shape deriving in
        # inference (validation) postprocessing
        'instance_fullres': forgeground_mask.clone()
    }
    data = (heatmap, offset), None    # None: no side outputs
    result_dict = post.postprocess(data, batch, is_training=False)
    found_instances = result_dict['instance_segmentation_gt_foreground']

    # show results
    img = (found_instances[0, 0] * int(255/found_instances[0, 0].max())).numpy()
    img_ref = (instances[0, 0] * int(255/instances[0, 0].max())).numpy()
    show_results(img=img, img_ref=img_ref)

    for n_batch in range(found_instances.shape[0]):
        for instance_id in found_instances[n_batch].unique():
            if instance_id == 0:
                continue
            instance_mask = found_instances[n_batch] == instance_id

            gt_num = instances[n_batch, instance_mask].unique()
            assert len(gt_num) == 1
            instances[n_batch, instance_mask] -= gt_num[0]

        assert instances[n_batch].sum() == 0
