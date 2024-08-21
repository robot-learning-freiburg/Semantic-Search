# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>

"""
from typing import List, Dict, Tuple

from nicr_scene_analysis_datasets.dataset_base import OrientationDict
import torch
from torchmetrics import Metric

from .pq import compare_and_accumulate
from .pq import PanopticQuality


def abs_angle_error_rad(
    pred_angle: torch.Tensor,
    target_angle: torch.Tensor
) -> torch.Tensor:
    pi = torch.pi

    # ensure that both are in [0, 2pi]
    pred_angle = pred_angle % (2*pi)
    target_angle = target_angle % (2*pi)

    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    angle_diff = pred_angle - target_angle
    angle_error = (angle_diff + pi) % (2*pi) - pi    # error is in [-pi, pi]

    return torch.abs(angle_error)


class MeanAbsoluteAngularError(Metric):
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state('sum_angular_error',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
        self.add_state('n_elements',
                       default=torch.tensor(0, dtype=torch.int64),
                       dist_reduce_fx='sum')

    def update(
        self,
        orientation_preds: List[OrientationDict],
        orientation_target: List[OrientationDict],
    ) -> None:
        for batch_idx in range(len(orientation_preds)):
            for pred_key, pred_angle in orientation_preds[batch_idx].items():
                target_angle = orientation_target[batch_idx][pred_key]

                self.sum_angular_error += abs_angle_error_rad(
                    pred_angle=torch.tensor(pred_angle),
                    target_angle=torch.tensor(target_angle),
                )
                self.n_elements += 1

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rad_error = self.sum_angular_error / self.n_elements
        deg_error = torch.rad2deg(rad_error)
        return rad_error, deg_error


class PanopticQualityWithOrientationMAE(PanopticQuality):
    full_state_update = False

    def __init__(self, *args, **kwargs):
        """
        Computes Panoptic Quality and Mean Absolute Angular Error for matched
        instances
        """
        super().__init__(*args, **kwargs)

        self.add_state('sum_angular_error',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
        self.add_state('n_elements',
                       default=torch.tensor(0, dtype=torch.int64),
                       dist_reduce_fx='sum')

    def update(self,
               panoptic_preds: torch.Tensor,
               orientation_preds: List[OrientationDict],
               panoptic_preds_id_dicts: List[Dict],
               panoptic_target: torch.Tensor,
               orientation_target: List[OrientationDict],
               panoptic_target_id_dicts: List[Dict]):

        assert panoptic_preds.ndim == 3
        assert len(panoptic_target) == len(panoptic_preds)
        for target, pred in zip(panoptic_target, panoptic_preds):
            assert target.shape == pred.shape

        # Important!!! The max value in instance_mask must be less then
        # max instance id. Else in _naively_combine_labels it will overlow
        # to the next semantic id.
        processes = []
        for pred, target in zip(panoptic_preds, panoptic_target):
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

        for batch_id, p in enumerate(processes):
            iou_per_class, tp_per_class, fn_per_class, fp_per_class, \
                matching = p.get()
            self.iou_per_class += iou_per_class
            self.tp_per_class += tp_per_class
            self.fn_per_class += fn_per_class
            self.fp_per_class += fp_per_class
            if orientation_preds is not None and orientation_target is not None:
                self.update_mae(orientation_preds[batch_id],
                                panoptic_preds_id_dicts[batch_id],
                                orientation_target[batch_id],
                                panoptic_target_id_dicts[batch_id],
                                matching)

    def update_mae(self,
                   orientation_preds: List[OrientationDict],
                   panoptic_preds_id_dicts: Dict,
                   orientation_target: List[OrientationDict],
                   panoptic_target_id_dicts: Dict,
                   matching: List[Tuple[int, int]]):

        for target_id, pred_id in matching:
            # Target with id 0 is not a real instance (stuff/void/background)
            if target_id == 0:
                continue

            # We can only calculate the MAE for instances with a orientation
            # in the target.
            if target_id not in panoptic_target_id_dicts:
                continue
            target_instance_id = panoptic_target_id_dicts[target_id]
            if target_instance_id not in orientation_target:
                continue
            target_angle = orientation_target[target_instance_id]

            # We cant have a orientation with id 0 in the prediction dict
            if pred_id not in panoptic_preds_id_dicts:
                continue
            pred_instance_id = panoptic_preds_id_dicts[pred_id]
            if pred_instance_id not in orientation_preds:
                continue
            pred_angle = orientation_preds[pred_instance_id]

            self.sum_angular_error += abs_angle_error_rad(
                pred_angle=torch.tensor(pred_angle),
                target_angle=torch.tensor(target_angle),
            )
            self.n_elements += 1

    def compute(self, suffix: str = '') -> Dict:
        r_dict = super().compute(suffix=suffix)

        rad_error = self.sum_angular_error / self.n_elements
        deg_error = torch.rad2deg(rad_error)
        r_dict[f'mae{suffix}_rad'] = rad_error
        r_dict[f'mae{suffix}_deg'] = deg_error

        return r_dict
