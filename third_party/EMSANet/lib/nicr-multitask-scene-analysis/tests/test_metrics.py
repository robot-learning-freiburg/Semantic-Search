# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Some parts of this code are based on:
    https://github.com/tensorflow/models/blob/v2.7.0/official/vision/beta/evaluation/panoptic_quality_test.py
and:
    https://github.com/tensorflow/models/blob/v2.7.0/official/vision/beta/evaluation/panoptic_quality_evaluator_test.py
"""
import tarfile
import pytest
import torch
from torchvision.transforms import Compose
import numpy as np
from tqdm import tqdm

from nicr_mt_scene_analysis import metric
from nicr_mt_scene_analysis.data import preprocessing
from nicr_mt_scene_analysis.data import RandomSamplerSubset
from nicr_mt_scene_analysis.model.postprocessing import get_postprocessing_class
from nicr_mt_scene_analysis.testing.dataset import get_dataset
from nicr_mt_scene_analysis.testing.dataset import get_dataloader
from nicr_mt_scene_analysis.utils.panoptic_merge import deeplab_merge_batch


def get_dataloader_for_tasks(tasks, batch_size=24, name='nyuv2',
                             subsample=True):
    sample_keys = ('semantic', 'instance', 'orientations', 'scene', 'rgb')
    dataset = get_dataset(
        name=name,
        split='test',
        sample_keys=sample_keys
    )
    dataset_config = dataset.config

    transforms = []
    if 'instance' in tasks:
        transforms.append(preprocessing.InstanceClearStuffIDs(
            semantic_classes_is_thing=dataset_config.semantic_label_list.classes_is_thing
        ))

    transforms.append(preprocessing.FullResCloner(
        keys_to_keep_fullres=sample_keys
    ))

    transforms.append(preprocessing.Resize(
        height=480,
        width=640
    ))

    if 'instance' in tasks:
        transforms.append(
            preprocessing.InstanceTargetGenerator(
                sigma=8,
                semantic_classes_is_thing=dataset_config.semantic_label_list.classes_is_thing
            )
        )
    transforms.append(preprocessing.ToTorchTensors())
    dataset.preprocessor = Compose(transforms)

    # we use a random subsample of 10% to speed up tests
    if subsample:
        sampler = RandomSamplerSubset(
            data_source=dataset,
            subset=0.2,
            deterministic=False
        )
        kwargs = {'sampler': sampler}
    else:
        kwargs = {}

    return get_dataloader(dataset, batch_size=batch_size, **kwargs)


def test_pq_perfect_match():
    device = torch.device('cpu')
    category_mask = torch.zeros((6, 6), dtype=torch.int32, device=device)
    instance_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 1, 1, 1],
        [1, 2, 1, 1, 1, 1],
    ], dtype=torch.int32, device=device)

    # Add batch axis
    category_mask.unsqueeze_(0)
    instance_mask.unsqueeze_(0)

    pq_metric = metric.PanopticQuality(
        num_categories=1,
        ignored_label=2,
        max_instances_per_category=16,
        offset=16,
        is_thing=torch.tensor([True]),
    )

    groundtruths = category_mask.long()*16+instance_mask.long()
    predictions = category_mask.long()*16+instance_mask.long()

    pq_metric.update(predictions, groundtruths)

    np.testing.assert_array_equal(pq_metric.iou_per_class, [2.0])
    np.testing.assert_array_equal(pq_metric.tp_per_class, [2])
    np.testing.assert_array_equal(pq_metric.fn_per_class, [0])
    np.testing.assert_array_equal(pq_metric.fp_per_class, [0])
    results = pq_metric.compute()
    np.testing.assert_array_equal(results['pq_per_class'], [1.0])
    np.testing.assert_array_equal(results['rq_per_class'], [1.0])
    np.testing.assert_array_equal(results['sq_per_class'], [1.0])
    assert results['all_pq'] == 1.0
    assert results['all_rq'] == 1.0
    assert results['all_sq'] == 1.0
    assert results['all_num_categories'] == 1


def test_pq_totally_wrong():
    device = torch.device('cpu')
    category_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.int32, device=device)
    instance_mask = torch.zeros([6, 6], dtype=torch.int32, device=device)

    # Add batch axis
    category_mask.unsqueeze_(0)
    instance_mask.unsqueeze_(0)

    groundtruths = category_mask.long()+instance_mask.long()
    predictions = (1-category_mask.long())+instance_mask.long()

    pq_metric = metric.PanopticQuality(
        num_categories=2,
        ignored_label=2,
        max_instances_per_category=1,
        offset=16,
        is_thing=torch.tensor([True, True]),)

    pq_metric.update(predictions, groundtruths)
    np.testing.assert_array_equal(pq_metric.iou_per_class,
                                  [0.0, 0.0])
    np.testing.assert_array_equal(pq_metric.tp_per_class,
                                  [0, 0])
    np.testing.assert_array_equal(pq_metric.fn_per_class,
                                  [1, 1])
    np.testing.assert_array_equal(pq_metric.fp_per_class,
                                  [1, 1])
    results = pq_metric.compute()

    np.testing.assert_array_equal(results['pq_per_class'], [0.0, 0.0])
    np.testing.assert_array_equal(results['rq_per_class'], [0.0, 0.0])
    np.testing.assert_array_equal(results['sq_per_class'], [0.0, 0.0])
    assert results['all_pq'] == 0.0
    assert results['all_rq'] == 0.0
    assert results['all_sq'] == 0.0
    assert results['all_num_categories'] == 2


def test_pq_matches_by_iou():
    device = torch.device('cpu')
    groundtruth_instance_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.int32, device=device)

    good_det_instance_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.int32, device=device)

    # Create groundtruths
    category_mask = torch.zeros_like(groundtruth_instance_mask)
    # Add batch axis
    category_mask.unsqueeze_(0)
    groundtruth_instance_mask.unsqueeze_(0)

    groundtruths = category_mask.long()*16+groundtruth_instance_mask.long()

    # Create predictions
    category_mask = torch.zeros_like(good_det_instance_mask)
    category_mask.unsqueeze_(0)
    good_det_instance_mask.unsqueeze_(0)
    predictions = category_mask.long()*16+good_det_instance_mask.long()

    pq_metric = metric.PanopticQuality(
        num_categories=1,
        ignored_label=2,
        max_instances_per_category=16,
        offset=16,
        is_thing=torch.tensor([True]))
    pq_metric.update(predictions, groundtruths)

    # iou(1, 1) = 28/30
    # iou(2, 2) = 6 / 8
    np.testing.assert_array_almost_equal(pq_metric.iou_per_class,
                                         [28 / 30 + 6 / 8])
    np.testing.assert_array_equal(pq_metric.tp_per_class, [2])
    np.testing.assert_array_equal(pq_metric.fn_per_class, [0])
    np.testing.assert_array_equal(pq_metric.fp_per_class, [0])
    results = pq_metric.compute()
    np.testing.assert_array_equal(results['pq_per_class'],
                                  [(28 / 30 + 6 / 8) / 2])
    np.testing.assert_array_equal(results['rq_per_class'], [1.0])
    np.testing.assert_array_equal(results['sq_per_class'],
                                  [(28 / 30 + 6 / 8) / 2])
    assert results['all_pq'] == (28 / 30 + 6 / 8) / 2
    assert results['all_rq'] == 1.0
    assert results['all_sq'] == (28 / 30 + 6 / 8) / 2
    assert results['all_num_categories'] == 1

    # Change data
    bad_det_instance_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.int32, device=device)
    bad_det_instance_mask.unsqueeze_(0)
    predictions = category_mask.long()*16+bad_det_instance_mask.long()

    pq_metric.reset()
    pq_metric.update(predictions, groundtruths)

    # iou(1, 1) = 27/32
    np.testing.assert_array_almost_equal(pq_metric.iou_per_class,
                                         [27 / 32])
    np.testing.assert_array_equal(pq_metric.tp_per_class, [1])
    np.testing.assert_array_equal(pq_metric.fn_per_class, [1])
    np.testing.assert_array_equal(pq_metric.fp_per_class, [1])
    results = pq_metric.compute()
    np.testing.assert_array_equal(results['pq_per_class'], [27 / 32 / 2])
    np.testing.assert_array_equal(results['rq_per_class'], [0.5])
    np.testing.assert_array_equal(results['sq_per_class'], [27 / 32])
    assert results['all_pq'] == 27 / 32 / 2
    assert results['all_rq'] == 0.5
    assert results['all_sq'] == 27 / 32
    assert results['all_num_categories'] == 1


def test_pq_wrong_instances():
    device = torch.device('cpu')
    category_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 1, 2, 2],
        [1, 2, 2, 1, 2, 2],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=torch.int32, device=device)

    groundtruth_instance_mask = torch.zeros([6, 6], dtype=torch.int32, device=device)
    predicted_instance_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.int32, device=device)

    category_mask.unsqueeze_(0)
    groundtruth_instance_mask.unsqueeze_(0)
    predicted_instance_mask.unsqueeze_(0)

    predictions = category_mask.long()*10+predicted_instance_mask.long()
    groundtruths = category_mask.long()*10+groundtruth_instance_mask.long()

    pq_metric = metric.PanopticQuality(
        num_categories=3,
        ignored_label=0,
        max_instances_per_category=10,
        offset=100,
        is_thing=[True, True, True])

    pq_metric.update(predictions, groundtruths)

    np.testing.assert_array_equal(pq_metric.iou_per_class,
                                  [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(pq_metric.tp_per_class,
                                  [0, 1, 0])
    np.testing.assert_array_equal(pq_metric.fn_per_class,
                                  [0, 0, 1])
    np.testing.assert_array_equal(pq_metric.fp_per_class,
                                  [0, 0, 2])
    results = pq_metric.compute()
    np.testing.assert_array_equal(results['pq_per_class'], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(results['rq_per_class'], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(results['sq_per_class'], [0.0, 1.0, 0.0])
    assert results['all_pq'] == 0.5
    assert results['all_rq'] == 0.5
    assert results['all_sq'] == 0.5
    assert results['all_num_categories'] == 2


def test_pq_instance_order_is_arbitrary():
    device = torch.device('cpu')
    category_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 1, 2, 2],
        [1, 2, 2, 1, 2, 2],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=torch.int32, device=device)

    groundtruth_instance_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.int32, device=device)

    predicted_instance_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.int32, device=device)

    category_mask.unsqueeze_(0)
    groundtruth_instance_mask.unsqueeze_(0)
    predicted_instance_mask.unsqueeze_(0)

    predictions = category_mask.long()*10+predicted_instance_mask.long()
    groundtruths = category_mask.long()*10+groundtruth_instance_mask.long()

    pq_metric = metric.PanopticQuality(
        num_categories=3,
        ignored_label=0,
        max_instances_per_category=10,
        offset=100,
        is_thing=[True, True, True])
    pq_metric.update(predictions, groundtruths)

    np.testing.assert_array_equal(pq_metric.iou_per_class,
                                  [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(pq_metric.tp_per_class,
                                  [0, 1, 2])
    np.testing.assert_array_equal(pq_metric.fn_per_class,
                                  [0, 0, 0])
    np.testing.assert_array_equal(pq_metric.fp_per_class,
                                  [0, 0, 0])
    results = pq_metric.compute()
    np.testing.assert_array_equal(results['pq_per_class'], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(results['rq_per_class'], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(results['sq_per_class'], [0.0, 1.0, 1.0])
    assert results['all_pq'] == 1.0
    assert results['all_rq'] == 1.0
    assert results['all_sq'] == 1.0
    assert results['all_num_categories'] == 2


def test_pq_multiple_batches():
    device = torch.device('cpu')
    category_mask = torch.zeros([6, 6], dtype=torch.int32, device=device)
    groundtruth_instance_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=torch.int32, device=device)

    good_det_instance_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=torch.int32, device=device)

    category_mask.unsqueeze_(0)
    good_det_instance_mask.unsqueeze_(0)
    groundtruth_instance_mask.unsqueeze_(0)

    predictions = category_mask.long()*16+good_det_instance_mask.long()
    groundtruths = category_mask.long()*16+groundtruth_instance_mask.long()

    pq_evaluator = metric.PanopticQuality(
        num_categories=1,
        ignored_label=2,
        max_instances_per_category=16,
        offset=16,
        is_thing=[True])

    # Batch size 2
    predictions = torch.cat((predictions, predictions))
    groundtruths = torch.cat((groundtruths, groundtruths))

    pq_evaluator.update(groundtruths, predictions)

    bad_det_instance_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 1],
        [1, 1, 1, 2, 2, 1],
        [1, 1, 1, 2, 2, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=torch.int32, device=device)

    bad_det_instance_mask.unsqueeze_(0)

    predictions = category_mask.long()*16+bad_det_instance_mask.long()
    predictions = torch.cat((predictions, predictions))

    pq_evaluator.update(groundtruths, predictions)

    results = pq_evaluator.compute()
    np.testing.assert_array_equal(results['pq_per_class'],
                                  [((28 / 30 + 6 / 8) + (27 / 32)) / 2 / 2])
    np.testing.assert_array_equal(results['rq_per_class'],
                                  [3 / 4])
    np.testing.assert_array_equal(results['sq_per_class'],
                                  [((28 / 30 + 6 / 8) + (27 / 32)) / 3])
    np.testing.assert_almost_equal(results['all_pq'], 0.63177083)
    assert results['all_rq'] == 0.75
    np.testing.assert_almost_equal(results['all_sq'], 0.84236111)
    assert results['all_num_categories'] == 1


def test_panoptic_quality_only():
    dataloader = get_dataloader_for_tasks(('semantic', 'instance'))
    dataset_config = dataloader.dataset.config
    n_semantic_classes = len(dataset_config.semantic_label_list)
    is_thing = dataset_config.semantic_label_list.classes_is_thing

    pq_metric = metric.PanopticQuality(
        num_categories=n_semantic_classes,
        ignored_label=0,
        max_instances_per_category=256,
        offset=256**3,
        is_thing=is_thing
    )

    for idx, batch in enumerate(tqdm(dataloader)):
        if idx > 20:
            break
        semantic_batch = batch['semantic']
        instance_batch = batch['instance']

        panoptic_targets = semantic_batch.long()*256+instance_batch.long()
        panoptic_preds = semantic_batch.long()*256+instance_batch.long()

        pq_metric.update(panoptic_preds, panoptic_targets)

    result = pq_metric.compute()
    assert result['things_pq'] == 1.0
    assert result['things_rq'] == 1.0
    assert result['things_sq'] == 1.0
    assert result['things_num_categories'] == float(sum(is_thing))

    assert result['stuff_pq'] == 1.0
    assert result['stuff_rq'] == 1.0
    assert result['stuff_sq'] == 1.0
    assert result['stuff_num_categories'] == n_semantic_classes-1-float(sum(is_thing))

    assert result['all_pq'] == 1.0
    assert result['all_rq'] == 1.0
    assert result['all_sq'] == 1.0
    # Void is ignored
    assert result['all_num_categories'] == n_semantic_classes - 1
    # Assert that there was no void match
    assert pq_metric.iou_per_class[0] == 0


def test_panoptic_quality():
    # TODO: Using subsampling for the last test would be better.
    # However, this can change the number of semantic classes, which
    # are processed by the PQ Metric.
    # At the end of this test, it is checked if all classes from the dataset
    # are found by the PQ Metric.
    # If a semantic class wasn not part of the test, it will fail.
    # A better solution would be to compare the number from PQ against the
    # real number of semantic classes which were part of testing.
    # By doing so, the subsample could be set to True again.
    dataloader = get_dataloader_for_tasks(('semantic', 'instance'),
                                          subsample=False)

    dataset_config = dataloader.dataset.config
    n_semantic_classes = len(dataset_config.semantic_label_list)
    is_thing = dataset_config.semantic_label_list_without_void.classes_is_thing

    semantic_postprocessing = get_postprocessing_class('semantic')()
    instance_postprocessing = get_postprocessing_class('instance')()
    postprocessing = get_postprocessing_class('panoptic')
    postprocessing = postprocessing(
        semantic_postprocessing=semantic_postprocessing,
        instance_postprocessing=instance_postprocessing,
        semantic_classes_is_thing=is_thing,
        semantic_class_has_orientation=is_thing,
    )

    is_thing = dataset_config.semantic_label_list.classes_is_thing
    pq_metric = metric.PanopticQuality(
        num_categories=n_semantic_classes,
        ignored_label=0,
        max_instances_per_category=256,
        offset=256**3,
        is_thing=is_thing
    )

    for batch in tqdm(dataloader):
        semantic_batch = batch['semantic']
        instance_batch = batch['instance']
        n, h, w = semantic_batch.shape
        # Output of network is none sparse
        semantic_output = torch.zeros((n, n_semantic_classes, h, w),
                                      dtype=torch.float32)
        for n_class in range(n_semantic_classes):
            mask = semantic_batch == (n_class+1)
            semantic_output[:, n_class] = mask.squeeze()

        # Prepare data for postprocessing
        # In batch the shape of the heatmap is (n, h, w) however the
        # network output is (n, 1, h, w).
        # As the postprocessing expects the heatmap to be (n, 1, h, w)
        # we need to unsqueeze the heatmap.
        instance_center = batch['instance_center'].unsqueeze(1).float()
        instance_offset = batch['instance_offset'].float()

        instance_output = (instance_center, instance_offset)
        side_outputs = (None, None)
        output = (semantic_output, instance_output)
        data = (output, side_outputs)
        result = postprocessing.postprocess(data, batch, is_training=False)
        segmentation_result = result["semantic_segmentation_idx_fullres"]
        # The network is not able to predict the void label.
        # Void label is idx=0 so we increase of all other labels by 1.
        segmentation_result += 1
        # In the evaluation the void label is ignored.
        # In mIoU Calaculation it get's masked out.
        # For computing the PQ/SQ/RQ masking out is done by the metric class.
        # All pixels with the void label are ignored.
        # Thats why we set all pixels to 0 which are void in the ground truth.
        void_mask = semantic_batch == 0
        segmentation_result[void_mask] = 0

        # Verify output of semantic segmentation
        assert (semantic_batch == segmentation_result).all()

        # Verify output of instance segmentation.
        # This isn't a trivial task and would require matching of the instance
        # ids. As a quick test we just verify that the number of instances is
        # correct.
        instance_result = result["panoptic_segmentation_deeplab_instance_idx"]
        # Note: Instance ids aren't unique over the batch axis.
        # Thats why we need to iterate over the axis.
        for batch_idx in range(n):
            current_instance_gt = instance_batch[batch_idx]
            current_fg = batch["instance_foreground"][batch_idx].bool()
            instance_batch_fg = current_instance_gt[current_fg]
            num_unique_gt_instances = len(instance_batch_fg.unique())
            num_unique_pred_instances = len(instance_result[batch_idx][current_fg].unique())
            num_center_points = (batch["instance_center"][batch_idx] == 1.0).sum()

            assert abs(num_unique_pred_instances - num_center_points) < 2
            # It's not allways possible to find all instances.
            # If instances have the same center of mass then the postprocessing
            # isn't able to differentiate them.
            # Thats why we check that the number of unique instances is almost
            # equal.
            assert abs(num_unique_gt_instances - num_unique_pred_instances) < 2

        panoptic_targets, _ = deeplab_merge_batch(semantic_batch,
                                                  instance_batch,
                                                  batch["instance_foreground"],
                                                  256,
                                                  np.where(is_thing)[0],
                                                  0)

        panoptic_preds, _ = deeplab_merge_batch(segmentation_result,
                                                instance_result,
                                                batch["instance_foreground"],
                                                256,
                                                np.where(is_thing)[0],
                                                0)

        pq_metric.update(panoptic_preds, panoptic_targets)

    result = pq_metric.compute()
    assert result['things_pq'] == 1.0
    assert result['things_rq'] == 1.0
    assert result['things_sq'] == 1.0
    assert result['things_num_categories'] == float(sum(is_thing))

    assert result['stuff_pq'] == 1.0
    assert result['stuff_rq'] == 1.0
    assert result['stuff_sq'] == 1.0
    assert result['stuff_num_categories'] == n_semantic_classes-1-float(sum(is_thing))

    assert result['all_pq'] == 1.0
    assert result['all_rq'] == 1.0
    assert result['all_sq'] == 1.0
    assert result['all_num_categories'] == n_semantic_classes-1
    # Assert that there was no void match
    assert pq_metric.iou_per_class[0] == 0


@pytest.mark.parametrize('mode', ('min', 'max'))
def test_mean_absolut_angular_error(mode):
    mae_metric = metric.mae.MeanAbsoluteAngularError()
    # postprocessing = get_postprocessing_class('orientations')

    target_result = 0.0
    # Lowest MAE is 180.0 (highest error) and highest is 0.0
    if mode == "min":
        target_result = 180.0
        target_result_rad = np.deg2rad(target_result)

    for _ in range(100):
        # Generate dummy data
        # Range is [-4*pi, 4*pi] to check if edge cases a handled correctly
        random_angles = torch.rand(100) * 8*torch.pi - 4*torch.pi
        # Move to dict
        target_dict = {}
        pred_dict = {}
        for name, angle in enumerate(random_angles):
            target_dict[name] = angle
            if mode == "min":
                rand = torch.rand(1)[0]
                if rand < 0.5:
                    angle = angle - target_result_rad
                else:
                    angle = angle + target_result_rad
            pred_dict[name] = angle

            mae_metric.update([target_dict], [pred_dict])

    rad_err, deg_err = mae_metric.compute()
    rad_err = rad_err.cpu().numpy()
    rad_err_to_deg = np.rad2deg(rad_err)
    rad_err_to_deg = float(rad_err_to_deg)
    deg_err = deg_err.cpu().numpy()
    deg_err = float(deg_err)
    target_result = float(target_result)
    np.testing.assert_almost_equal(rad_err_to_deg, deg_err, decimal=5)
    np.testing.assert_almost_equal(deg_err, target_result, decimal=5)


@pytest.mark.parametrize('mode', ('min', 'max'))
@pytest.mark.parametrize('name', ('nyuv2', 'sunrgbd'))
def test_mean_absolut_angular_error_data(mode, name):
    dataloader = get_dataset(sample_keys=('orientations', 'instance'),
                             split='test',
                             name=name)
    mae_metric = metric.mae.MeanAbsoluteAngularError()
    # postprocessing = get_postprocessing_class('orientations')

    target_result = 0.0
    # Lowest MAE is 180.0 (highest error) and highest is 0.0
    if mode == "min":
        target_result = 180.0
        target_result_rad = np.deg2rad(target_result)

    for batch in dataloader:
        gt = batch['orientations']
        pred = batch['orientations']
        if mode == "min":
            pred_tmp = pred
            pred = {}
            for key, value in pred_tmp.items():
                rand = torch.rand(1)[0]
                if rand < 0.5:
                    value = value - target_result_rad
                else:
                    value = value + target_result_rad
                pred[key] = value
        mae_metric.update([gt], [pred])

    rad_err, deg_err = mae_metric.compute()
    rad_err = rad_err.cpu().numpy()
    rad_err_to_deg = np.rad2deg(rad_err)
    rad_err_to_deg = float(rad_err_to_deg)
    deg_err = deg_err.cpu().numpy()
    deg_err = float(deg_err)
    target_result = float(target_result)
    np.testing.assert_almost_equal(rad_err_to_deg, deg_err, decimal=5)
    np.testing.assert_almost_equal(deg_err, target_result, decimal=5)


@pytest.mark.parametrize('masked', (False, True))
def test_root_mean_squared_error_perfect(masked):
    device = torch.device('cpu')
    batch_size = 2
    height = 4
    width = 5

    # create targets and predictions (both are equal -> perfect metric)
    target = torch.testing.make_tensor(
        (batch_size, 3, height, width),
        low=-1,
        high=1,
        dtype=torch.float,
        device=device
    )
    preds = target.clone()

    # create random mask to select valid values
    if masked:
        mask = torch.testing.make_tensor(
            (batch_size, height, width),
            dtype=torch.bool,
            device=device
        )
    else:
        mask = None

    # compute metric
    rmse_metric = metric.RootMeanSquaredError()
    rmse_metric.to(device)

    rmse_metric.update(preds, target, mask)
    rmse = rmse_metric.compute()

    # check result
    rmse_expected = torch.tensor(0.0, dtype=torch.float32, device=device)
    torch.testing.assert_close(rmse, rmse_expected)


@pytest.mark.parametrize('masked', (False, True))
def test_root_mean_squared_error_totally_wrong(masked):
    device = torch.device('cpu')
    batch_size = 2
    height = 4
    width = 5

    # create targets (all elements are equal to (1, 2, 3)) and
    # predictions (all elements are equal to (-1, -2, -3))
    target = torch.ones(
        size=(batch_size, 3, height, width),
        dtype=torch.float32,
        device=device
    )
    target[:, 1, :, :] += 1
    target[:, 2, :, :] += 2
    preds = -target

    # create random mask to select valid values
    if masked:
        mask = torch.testing.make_tensor(
            (batch_size, height, width),
            dtype=torch.bool,
            device=device
        )
    else:
        mask = None

    # compute metric
    rmse_metric = metric.RootMeanSquaredError()
    rmse_metric.to(device)

    rmse_metric.update(preds, target, mask)
    rmse = rmse_metric.compute()

    # check result
    squared_error = (np.array((1, 2, 3)) - np.array((-1, -2, -3))) ** 2
    mse_expected = torch.tensor(np.mean(squared_error, dtype='float32'),
                                device=device, dtype=torch.float32)
    rmse_expected = torch.sqrt(mse_expected)

    torch.testing.assert_close(rmse, rmse_expected)
