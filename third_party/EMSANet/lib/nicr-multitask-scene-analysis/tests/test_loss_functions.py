# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import pytest
import torch

from nicr_mt_scene_analysis.loss import CrossEntropyLossSemantic
from nicr_mt_scene_analysis.loss import L1Loss
from nicr_mt_scene_analysis.loss import MSELoss
from nicr_mt_scene_analysis.loss import VonMisesLossBiternion


def get_tensors(input_shape, target_shape, use_scales=False):
    assert input_shape[-2:] == target_shape[-2:]

    input_tensor = torch.rand(input_shape, dtype=torch.float32)
    target_tensor = torch.rand(target_shape, dtype=torch.float32)

    input_tensors = [input_tensor]
    target_tensors = [target_tensor]

    if use_scales:
        # add additional scaled side outputs
        h, w = input_shape[-2:]
        n_scales = 3

        for exp in range(n_scales):
            scale = 2**(exp+1)
            target_tensors.append(target_tensor[..., :h//scale, :w//scale])
            input_tensors.append(input_tensor[..., :h//scale, :w//scale])

    return tuple(input_tensors), tuple(target_tensors)


@pytest.mark.parametrize('batch_size', (1, 8))
@pytest.mark.parametrize('use_weights', (False, True))
@pytest.mark.parametrize('use_scales', (False, True))
@pytest.mark.parametrize('label_smoothing', (0.0, 0.5))
def test_ce(batch_size, use_weights, use_scales, label_smoothing):
    # get inputs and targets
    h, w = 480, 640
    n_classes = 40
    input_tensors, target_tensors = get_tensors(
        input_shape=(batch_size, n_classes, h, w),
        target_shape=(batch_size, h, w),
        use_scales=use_scales
    )
    target_tensors = [(t*n_classes).long() for t in target_tensors]

    # get loss function
    if use_weights:
        weights = torch.rand(n_classes, dtype=torch.float32)
    else:
        weights = None
    loss_function = CrossEntropyLossSemantic(
        weights=weights,
        label_smoothing=label_smoothing
    )

    # apply loss function
    loss_outputs = loss_function(input_tensors, target_tensors)

    # some simple checks
    assert len(target_tensors) == len(loss_outputs)
    for target, (loss, n_elements) in zip(target_tensors, loss_outputs):
        assert loss != 0
        assert n_elements == (target > 0).sum()


@pytest.mark.parametrize('batch_size', (1, 8))
@pytest.mark.parametrize('use_scales', (False, True))
@pytest.mark.parametrize('reduction', ('none', 'mean', 'sum'))
def test_l1(batch_size, use_scales, reduction):
    # get inputs and targets
    h, w = 480, 640
    input_tensors, target_tensors = get_tensors(
        input_shape=(batch_size, 2, h, w),
        target_shape=(batch_size, 2, h, w),
        use_scales=use_scales
    )

    # get and apply loss function
    loss_function = L1Loss(reduction)
    loss_outputs = loss_function(input_tensors, target_tensors)

    assert len(target_tensors) == len(loss_outputs)
    for i, (loss, n_elements) in enumerate(loss_outputs):
        if 'none' == reduction:
            assert loss.shape == input_tensors[i].shape
            assert n_elements == input_tensors[i].numel()
        elif 'mean' == reduction:
            assert loss.shape == ()
            assert loss != 0
            assert n_elements == 1
        elif 'sum' == reduction:
            assert loss.shape == ()
            assert loss != 0
            b, c, h_, w_ = input_tensors[i].shape
            assert n_elements == b * h_ * w_


@pytest.mark.parametrize('batch_size', (1, 8))
@pytest.mark.parametrize('use_scales', (False, True))
@pytest.mark.parametrize('reduction', ('none', 'mean', 'sum'))
def test_mse(batch_size, use_scales, reduction):
    # get inputs and targets
    h, w = 480, 640
    input_tensors, target_tensors = get_tensors(
        input_shape=(batch_size, 2, h, w),
        target_shape=(batch_size, 2, h, w),
        use_scales=use_scales
    )

    # get and apply loss function
    loss_function = MSELoss(reduction)
    loss_outputs = loss_function(input_tensors, target_tensors)

    assert len(target_tensors) == len(loss_outputs)
    for i, (loss, n_elements) in enumerate(loss_outputs):
        if 'none' == reduction:
            assert loss.shape == input_tensors[i].shape
            assert n_elements == input_tensors[i].numel()
        elif 'mean' == reduction:
            assert loss.shape == ()
            assert loss != 0
            assert n_elements == 1
        elif 'sum' == reduction:
            assert loss.shape == ()
            assert loss != 0
            b, c, h_, w_ = input_tensors[i].shape
            assert n_elements == b * h_ * w_


@pytest.mark.parametrize('batch_size', (1, 8))
@pytest.mark.parametrize('use_scales', (False, True))
@pytest.mark.parametrize('with_random_masks', (False, True))
def test_vonmises(batch_size, use_scales, with_random_masks):
    # get inputs and targets
    h, w = 480, 640
    input_tensors, target_tensors = get_tensors(
        input_shape=(batch_size, 2, h, w),
        target_shape=(batch_size, 2, h, w),
        use_scales=use_scales
    )

    # (b, c, h, w) -> (b, h, w, c) -> (b*h*w, c) with c = 2
    input_tensors = [t.permute(0, 2, 3, 1).reshape(-1, 2)
                     for t in input_tensors]
    target_tensors = [t.permute(0, 2, 3, 1).reshape(-1, 2)
                      for t in target_tensors]

    if with_random_masks:
        # apply some random masks
        masks = [torch.rand((inp.shape[0],)) > 0.5 for inp in input_tensors]
        input_tensors = [t[mask, :] for t, mask in zip(input_tensors, masks)]
        target_tensors = [t[mask, :] for t, mask in zip(target_tensors, masks)]

    # get and apply loss function
    loss_function = VonMisesLossBiternion()
    loss_outputs = loss_function(input_tensors, target_tensors)

    assert len(target_tensors) == len(loss_outputs)
    for i, (loss, n_elements) in enumerate(loss_outputs):
        assert loss != 0
        if with_random_masks:
            assert n_elements == masks[i].sum()
        else:
            assert n_elements == (input_tensors[i].numel() // 2)    # two channels
