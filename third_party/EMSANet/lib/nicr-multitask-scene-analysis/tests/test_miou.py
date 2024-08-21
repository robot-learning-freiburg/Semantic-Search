# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from time import perf_counter

import numpy as np

import pytest
import torch
from torchmetrics import JaccardIndex as IoU

# TODO: note that tests fail with MulticlassJaccardIndex in 0.10.2 and 0.11.0
# see: https://github.com/Lightning-AI/metrics/issues/1385
# from torchmetrics.classification import MulticlassJaccardIndex as IoU

from nicr_mt_scene_analysis.metric import MeanIntersectionOverUnion


def get_batch(batch_size, height, width, n_classes_without_void, device):
    # prediction (does not include void)
    pred = torch.rand((batch_size, n_classes_without_void, height, width))
    pred_idx = torch.argmax(pred, dim=1)

    # prediction (does include void)
    target = torch.rand((batch_size, height, width))
    target = (target * (n_classes_without_void + 1)).long()

    return pred_idx.to(device), target.to(device)


@pytest.mark.parametrize('n_classes_without_void', (5, 40, 100))
def test_torchmetrics_iou(n_classes_without_void):
    device = torch.device('cpu')
    # metric without void masked before
    metric = IoU(
        num_classes=n_classes_without_void + 1,
        ignore_index=0,
        reduction='elementwise_mean'
    ).to(device)

    # metric with void masked before
    metric_masked = IoU(
        num_classes=n_classes_without_void,
        reduction='elementwise_mean'
    ).to(device)

    # bugfix: confmat is initialized as type float32 (in v0.6.1) which
    # results in slightly varying ground truth counts over several epochs
    # due to accumulating large numbers as float32
    # see: https://github.com/PyTorchLightning/metrics/pull/715
    metric._defaults['confmat'] = metric._defaults['confmat'].long()
    metric.reset()
    metric_masked._defaults['confmat'] = metric_masked._defaults['confmat'].long()
    metric_masked.reset()

    # update metrics several times and time updating
    times = []
    times_masked = []

    for _ in range(10):
        preds, target = get_batch(batch_size=64, height=100, width=100,
                                  n_classes_without_void=n_classes_without_void,
                                  device=device)

        # update metric
        start = perf_counter()
        metric.update(preds + 1, target)
        times.append(perf_counter()-start)

        # update reference metric
        start = perf_counter()
        mask = target != 0
        metric_masked.update(preds[mask], target[mask]-1)
        times_masked.append(perf_counter()-start)

    miou = metric.compute()
    miou_masked = metric_masked.compute()

    print(f"miou: {miou}, miou_masked: {miou_masked}")
    assert miou == miou_masked

    # as of 01/08/2022:
    # - masking before is faster
    # - both version are VERY SLOW -> see MeanIntersectionOverUnion
    print(f"time miou: {np.mean(times)}, "
          f"time miou_masked: {np.mean(times_masked)}")


@pytest.mark.parametrize('n_classes_without_void', (5, 40, 100))
def test_own_miou(n_classes_without_void):
    device = torch.device('cpu')

    metric = MeanIntersectionOverUnion(
        n_classes=n_classes_without_void
    ).to(device)

    metric_with_void = MeanIntersectionOverUnion(
        n_classes=n_classes_without_void+1,
        ignore_first_class=True
    ).to(device)

    # reference metric
    metric_ref = IoU(
        num_classes=n_classes_without_void,
        reduction='elementwise_mean'
    ).to(device)

    # bugfix: confmat is initialized as type float32 (in v0.6.1) which
    # results in slightly varying ground truth counts over several epochs
    # due to accumulating large numbers as float32
    # see: https://github.com/PyTorchLightning/metrics/pull/715
    metric_ref._defaults['confmat'] = metric_ref._defaults['confmat'].long()
    metric_ref.reset()

    # update metrics several times and time updating
    times = []
    times_with_void = []
    times_ref = []

    for _ in range(10):
        preds, target = get_batch(batch_size=64, height=100, width=100,
                                  n_classes_without_void=n_classes_without_void,
                                  device=device)

        # update metric with void
        start = perf_counter()
        metric_with_void.update(preds+1, target)
        times_with_void.append(perf_counter()-start)

        # mask void
        mask = target != 0
        preds, target = preds[mask], target[mask]-1

        # update metric
        start = perf_counter()
        metric.update(preds, target)
        times.append(perf_counter()-start)

        # update reference metric
        start = perf_counter()
        metric_ref.update(preds, target)
        times_ref.append(perf_counter()-start)

    miou_with_void, ious_with_void = metric_with_void.compute(return_ious=True)
    miou, ious = metric.compute(return_ious=True)
    miou_ref = metric_ref.compute()

    print(f"miou: {miou}, miou_with_void: {miou_with_void}, "
          f"miou_torchmetrics: {miou_ref}")

    # check results
    assert miou == miou_with_void == miou_ref
    assert torch.isnan(ious_with_void[0])    # void is ignored, so iou is nan
    assert (ious_with_void[1:] == ious).all()   # remaining ious must be equal

    # as of 01/08/2022: MeanIntersectionOverUnion is much faster!
    print(f"time miou: {np.mean(times)}, "
          f"time miou_with_void: {np.mean(times_with_void)}, "
          f"time metric_torchmetrics: {np.mean(times_ref)}, "
          f"speedup : {np.mean(times_ref)/np.mean(times):4.2f}")
