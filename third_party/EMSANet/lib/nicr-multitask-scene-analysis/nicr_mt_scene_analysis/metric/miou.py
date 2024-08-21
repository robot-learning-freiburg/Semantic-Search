# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
from torchmetrics import Metric


class MeanIntersectionOverUnion(Metric):
    full_state_update = False

    def __init__(
        self,
        n_classes: int,
        ignore_first_class: bool = False
    ) -> None:
        super().__init__()

        # note that states are automatically reset to their defaults when
        # calling metric.reset()
        self.add_state(
            'confmat',
            default=torch.zeros((n_classes, n_classes), dtype=torch.int64),
            dist_reduce_fx='sum'
        )

        # determine dtype for bincounting later on
        n_classes_squared = n_classes**2
        if n_classes_squared < 2**(8-1)-1:
            self._dtype = torch.int8
        elif n_classes_squared < 2**(16-1)-1:
            self._dtype = torch.int16
        else:
            # it does not matter in our tests
            self._dtype = torch.int64    # equal to long

        self._n_classes = n_classes
        self._ignore_first_class = ignore_first_class

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # convert dtype to speed up bincounting
        preds_ = preds.to(self._dtype)
        target_ = target.to(self._dtype)

        # compute confusion matrix
        unique_mapping = (target_.view(-1) * self._n_classes + preds_.view(-1))
        cnts = torch.bincount(unique_mapping,
                              minlength=self._n_classes**2)
        confmat = cnts.reshape(self._n_classes, self._n_classes)

        # update internal confusion matrix
        self.confmat += confmat

    def compute(self, return_ious: bool = False) -> torch.Tensor:
        tp = torch.diag(self.confmat).float()
        sum_pred = torch.sum(self.confmat, dim=0).float()
        sum_gt = torch.sum(self.confmat, dim=1).float()

        # ignore first class (void)
        if self._ignore_first_class:
            tp = tp[1:]
            sum_pred = sum_pred[1:]
            sum_gt = sum_gt[1:]
            sum_pred -= self.confmat[0, 1:].float()

        # we do want ignore classes without gt pixels
        mask = sum_gt != 0
        tp = tp[mask]
        sum_pred = sum_pred[mask]
        sum_gt = sum_gt[mask]

        # compute iou(s)
        intersection = tp
        union = sum_pred + sum_gt - tp
        iou = intersection/union

        if return_ious:
            # additionally return iou for each class
            # we assign nan if there is no gt and for first class (void)) if
            # ignored
            ious = torch.full((self._n_classes,), torch.nan,
                              dtype=torch.float32)
            iou_idx = mask.nonzero(as_tuple=True)[0]
            if self._ignore_first_class:
                iou_idx += 1
            ious[iou_idx] = iou

            return torch.mean(iou), ious

        return torch.mean(iou)
