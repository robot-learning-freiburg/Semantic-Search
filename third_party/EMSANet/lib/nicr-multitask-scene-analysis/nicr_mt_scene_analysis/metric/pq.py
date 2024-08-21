# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>

The code is mostly based on the following implementation:
    https://github.com/tensorflow/models/blob/v2.7.0/official/vision/beta/evaluation/panoptic_quality.py

This PQ implementation has several advantages over the original one of the
panopticapi package (see: https://github.com/cocodataset/panopticapi)
Advantages:
- do not use buggy multiprocessing code
  --> see: (https://github.com/cocodataset/panopticapi/issues/27)
- Work on torch.Tensor instead of PIL images
- do not require file writes
- directly work on a semantic and instance segmentation instead of
  a combined panoptic segmentation
- implemented as a torchmetrics metric
"""
from typing import Dict, List, Union, Tuple

from torchmetrics import Metric
from torch import multiprocessing as mp
import torch


# EPSILON used for division by zero
_EPSILON = 1e-10


def _ids_to_counts(id_array: torch.Tensor) -> Dict[int, int]:
    ids, counts = torch.unique(id_array, return_counts=True)
    return {int(id_v): int(count) for (id_v, count) in zip(ids, counts)}


def prediction_void_overlap(
    pred_segment_id: int,
    void_segment_id: int,
    offset: int,
    intersection_areas: Dict[int, int]
) -> int:
    # Helper function that computes the area of the overlap between
    # a predicted segment and the ground-truth void/ignored segment.
    void_intersection_id = void_segment_id * offset + pred_segment_id
    return intersection_areas.get(void_intersection_id, 0)


def prediction_ignored_overlap(
    pred_segment_id: int,
    ignored_segment_ids: int,
    offset: int,
    intersection_areas: Dict[int, int]
) -> int:
    total_ignored_overlap = 0
    for ignored_segment_id in ignored_segment_ids:
        intersection_id = ignored_segment_id * offset + pred_segment_id
        total_ignored_overlap += intersection_areas.get(intersection_id, 0)
    return total_ignored_overlap


def compare_and_accumulate(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_categories: int,
    ignored_label: int,
    max_instances_per_category,
    offset: int,
    void_segment_id: int
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:

    iou_per_class = torch.zeros(num_categories,
                                dtype=torch.float64).share_memory_()

    tp_per_class = torch.zeros(num_categories,
                               dtype=torch.float64).share_memory_()

    fn_per_class = torch.zeros(num_categories,
                               dtype=torch.float64).share_memory_()

    fp_per_class = torch.zeros(num_categories,
                               dtype=torch.float64).share_memory_()

    # Pre-calculate areas for all groundtruth and predicted segments.
    target_segment_areas = _ids_to_counts(target)
    pred_segment_areas = _ids_to_counts(pred)

    # There may be other ignored groundtruth segments with instance
    # id > 0, find those ids using the unique segment ids extracted with
    # the area computation above.
    ignored_segment_ids = set()
    for target_segment_id in target_segment_areas:
        current_label = target_segment_id//max_instances_per_category
        if current_label == ignored_label:
            ignored_segment_ids.add(target_segment_id)

    # Next, combine the ground-truth and predicted labels. Dividing up
    # the pixels based on which ground-truth segment and which predicted
    # segment they belong to, this will assign a different 32-bit integer
    # label to each choice of (ground-truth segment, predicted segment),
    # encoded as
    #   target * offset + pred.
    # TODO: Usually that should be a uint64 or int128 which both do not
    # exists in pytorch. Currently there is no guarantee that this does not
    # overflow.
    intersection_id_array = target * offset + pred

    # For every combination of (ground-truth segment, predicted segment)
    # with a non-empty intersection, this counts the number of pixels
    # in that intersection.
    intersection_areas = _ids_to_counts(intersection_id_array)

    # Sets that are populated with which segments ground-truth/predicted
    # segments have been matched with overlapping predicted/ground-truth
    # segments respectively.
    gt_matched = set()
    pred_matched = set()
    matched_instances = set()

    # Calculate IoU per pair of intersecting segments of the same category.
    for intersection_id, intersection_area in intersection_areas.items():
        if intersection_id == void_segment_id:
            continue

        gt_segment_id = int(intersection_id // offset)
        pred_segment_id = int(intersection_id % offset)

        gt_category = int(gt_segment_id // max_instances_per_category)
        pred_category = int(pred_segment_id // max_instances_per_category)
        if gt_category != pred_category:
            continue

        # Union between the ground-truth and predicted segments being
        # compared does not include the portion of the predicted segment
        # that consists of ground-truth "void" pixels.
        r = prediction_void_overlap(pred_segment_id,
                                    void_segment_id,
                                    offset,
                                    intersection_areas)

        tsa = target_segment_areas[gt_segment_id]
        psa = pred_segment_areas[pred_segment_id]
        ia = intersection_area

        union = (tsa + psa - ia - r)

        iou = intersection_area / union

        if iou > 0.5:
            tp_per_class[gt_category] += 1
            iou_per_class[gt_category] += iou
            gt_matched.add(gt_segment_id)
            pred_matched.add(pred_segment_id)
            matched_instances.add((gt_segment_id, pred_segment_id))

    # Count false negatives for each category.
    for gt_segment_id in target_segment_areas:
        if gt_segment_id in gt_matched:
            continue
        category = gt_segment_id // max_instances_per_category
        # Failing to detect a void segment is not a false negative.
        if category == ignored_label:
            continue
        fn_per_class[category] += 1

    # Count false positives for each category.
    for pred_segment_id in pred_segment_areas:
        if pred_segment_id in pred_matched:
            continue
        # A false positive is not penalized if is mostly ignored in the
        # ground truth.
        pio = prediction_ignored_overlap(pred_segment_id,
                                         ignored_segment_ids,
                                         offset,
                                         intersection_areas)
        if (pio / pred_segment_areas[pred_segment_id]) > 0.5:
            continue
        category = pred_segment_id // max_instances_per_category
        fp_per_class[category] += 1

    return iou_per_class, tp_per_class, fn_per_class, fp_per_class, matched_instances


def realdiv_maybe_zero(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(
        torch.less(torch.abs(y), _EPSILON),
        torch.zeros_like(x),
        torch.divide(x, y)
    )


class PanopticQuality(Metric):
    full_state_update = False

    def __init__(
        self, num_categories: int,
        ignored_label: int,
        max_instances_per_category: int,
        offset: int,
        is_thing: Union[torch.Tensor, List[bool]],
        num_workers=None
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.ignored_label = ignored_label
        self.max_instances_per_category = max_instances_per_category
        self.offset = offset
        self.is_thing = torch.tensor(is_thing, dtype=torch.bool).share_memory_()
        self.is_stuff = torch.logical_not(self.is_thing).share_memory_()

        assert len(self.is_thing) == self.num_categories

        # The update function is not vectorized, so we use a multiprocessing
        # pool for parallelization.
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 32)

        # Using 'fork' as a start method does not work with CUDA.
        ctx = mp.get_context('spawn')
        self.workers = ctx.Pool(processes=num_workers)

        # We assume there is only one void segment and it has instance id = 0.
        void_segment_id = self.ignored_label * self.max_instances_per_category
        self.void_segment_id = void_segment_id

        # dist_reduce_fx="sum" should be right.
        # See: https://github.com/tensorflow/models/blob/master/research/deeplab/evaluation/panoptic_quality.py#L247
        # note that states are automatically reset to their defaults when
        # calling metric.reset()
        self.add_state('iou_per_class',
                       default=torch.zeros(self.num_categories,
                                           dtype=torch.float64),
                       dist_reduce_fx='sum')

        self.add_state('tp_per_class',
                       default=torch.zeros(self.num_categories,
                                           dtype=torch.float64),
                       dist_reduce_fx='sum')

        self.add_state('fn_per_class',
                       default=torch.zeros(self.num_categories,
                                           dtype=torch.float64),
                       dist_reduce_fx='sum')

        self.add_state('fp_per_class',
                       default=torch.zeros(self.num_categories,
                                           dtype=torch.float64),
                       dist_reduce_fx='sum')

    def __del__(self):
        self.workers.terminate()
        self.workers.close()
        self.workers.join()

    def _valid_categories(self) -> torch.Tensor:
        valid_categories = torch.not_equal(
            self.tp_per_class + self.fn_per_class + self.fp_per_class, 0)
        if self.ignored_label >= 0 and self.ignored_label < self.num_categories:
            valid_categories[self.ignored_label] = False
        return valid_categories

    def _valid_categories_with_gt(self) -> torch.Tensor:
        valid_categories = torch.not_equal(
            self.tp_per_class + self.fn_per_class, 0)
        if self.ignored_label >= 0 and self.ignored_label < self.num_categories:
            valid_categories[self.ignored_label] = False
        return valid_categories

    def update(self,
               preds: Union[List[torch.Tensor], torch.Tensor],
               targets: Union[List[torch.Tensor], torch.Tensor]):
        assert preds.ndim == 3
        assert targets.shape == preds.shape

        # Important!!! The max value in instance_mask must be less then
        # max instance id. Else in _naively_combine_labels it will overflow
        # to the next semantic id.
        processes = []
        for pred, target in zip(preds, targets):
            if pred.device.type == 'cuda':
                pred.share_memory_()
                target.share_memory_()
            p = self.workers.apply_async(compare_and_accumulate,
                                         (pred,
                                          target,
                                          self.num_categories,
                                          self.ignored_label,
                                          self.max_instances_per_category,
                                          self.offset,
                                          self.void_segment_id))
            processes.append(p)

        for p in processes:
            iou_per_class, tp_per_class, fn_per_class, fp_per_class, _ = p.get()
            self.iou_per_class += iou_per_class
            self.tp_per_class += tp_per_class
            self.fn_per_class += fn_per_class
            self.fp_per_class += fp_per_class

    def result_per_category(self) -> Dict:
        sq_per_class = realdiv_maybe_zero(self.iou_per_class,
                                          self.tp_per_class)
        rq_per_class = realdiv_maybe_zero(
            self.tp_per_class,
            self.tp_per_class + 0.5*self.fn_per_class + 0.5*self.fp_per_class)
        return {
            'sq_per_class': sq_per_class,
            'rq_per_class': rq_per_class,
            'pq_per_class': torch.multiply(sq_per_class, rq_per_class)
        }

    def compute(self, suffix: str = '') -> Dict:
        results = self.result_per_category()
        # valid categories (with FP in classes without GT=TP+FN)
        valid_categories = self._valid_categories()
        # valid categories (without FP in classes without GT=TP+FN)
        valid_categories_with_gt = self._valid_categories_with_gt()

        # break down which categories are valid _and_ things/stuff.
        category_sets = {
            # measures that follow official panopticapi
            f'all{suffix}': valid_categories,
            f'things{suffix}': torch.logical_and(
                valid_categories, self.is_thing
            ),
            f'stuff{suffix}': torch.logical_and(
                valid_categories,
                self.is_stuff
            ),
            # measures that ignore FP in classes without gt -> the number of
            # categories is fixed and cannot vary during training
            f'all_with_gt{suffix}': valid_categories_with_gt,
            f'things_with_gt{suffix}': torch.logical_and(
                valid_categories_with_gt,
                self.is_thing
            ),
            f'stuff_with_gt{suffix}': torch.logical_and(
                valid_categories_with_gt,
                self.is_stuff)
        }

        for category_set_name, in_category_set in category_sets.items():
            if torch.any(in_category_set):
                results.update({
                    f'{category_set_name}_pq':
                        torch.mean(results['pq_per_class'][in_category_set]),
                    f'{category_set_name}_sq':
                        torch.mean(results['sq_per_class'][in_category_set]),
                    f'{category_set_name}_rq':
                        torch.mean(results['rq_per_class'][in_category_set]),
                    # the number of categories in this subset.
                    f'{category_set_name}_num_categories':
                        torch.sum(in_category_set.int()),
                })
            else:
                results.update({
                    f'{category_set_name}_pq': torch.tensor(0),
                    f'{category_set_name}_sq': torch.tensor(0),
                    f'{category_set_name}_rq': torch.tensor(0),
                    f'{category_set_name}_num_categories': torch.tensor(0)
                })

        return results
