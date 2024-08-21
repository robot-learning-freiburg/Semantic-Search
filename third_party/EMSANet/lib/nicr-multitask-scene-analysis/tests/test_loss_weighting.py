# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from copy import deepcopy

import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch

from nicr_mt_scene_analysis.loss_weighting import DynamicWeightAverage
from nicr_mt_scene_analysis.loss_weighting import RandomLossWeighting
from nicr_mt_scene_analysis.loss_weighting import FixedLossWeighting


@pytest.mark.parametrize('temperature', (1, 2))
def test_dwa(temperature):
    """Test DynamicWeightAverage"""
    n_batches = 10
    losses_epoch = {
        0: {
            'loss0': torch.ones((n_batches,)) * 1,
            'loss1': torch.ones((n_batches,)) * 2,
            'additional_loss': torch.ones((n_batches,)) * 50
        },
        1: {
            'loss0': torch.ones((n_batches,)) * 3,
            'loss1': torch.ones((n_batches,)) * 4,
            'additional_loss': torch.ones((n_batches,)) * 51
        },
        2: {
            'loss0': torch.ones((n_batches,)) * 5,
            'loss1': torch.ones((n_batches,)) * 6,
            'additional_loss': torch.ones((n_batches,)) * 52
        },
    }

    # precompute weights for epoch 2 (3rd epoch)
    # epoch 0: mean loss1: 1, mean loss2: 2
    # epoch 1: mean loss1: 3, mean loss2: 4
    # loss history as t-1/t-2
    loss_history_with_temp = np.array([3./1., 4./2.]) / temperature
    softmax = lambda x: np.exp(x) / np.exp(x).sum()
    weights_epoch2 = {
        'loss0': 2 * softmax(loss_history_with_temp)[0],
        'loss1': 2 * softmax(loss_history_with_temp)[1],
    }

    # test
    dwa = DynamicWeightAverage(
        loss_keys_to_consider=('loss0', 'loss1'),
        temperature=temperature
    )

    # minic sanity check
    loss = dwa.reduce_losses(
        losses={k: losses_epoch[0][k][0] for k in losses_epoch[0]},
        batch_idx=0
    )
    assert loss == 3    # simple sum (weights are all equal to 1)

    dwa.reset_weights()
    assert len(dwa._loss_buffer) == 0
    assert len(dwa._loss_history) == 0

    # mimic training
    for epoch in range(3):
        for batch_idx in range(n_batches):

            # weights always must sum to number of tasks
            assert np.isclose(sum(dwa.weights.values()), 2)

            loss = dwa.reduce_losses(
                losses={k: losses_epoch[epoch][k][batch_idx]
                        for k in losses_epoch[epoch]},
                batch_idx=batch_idx
            )
            if 0 == epoch:
                assert all(list(v == 1 for v in dwa.weights.values()))
                assert loss == 1+2    # simple sum (weights are all equal to 1)
            elif 1 == epoch:
                assert all(list(v == 1 for v in dwa.weights.values()))
                assert loss == 3+4    # simple sum (weights are all equal to 1)
            elif 2 == epoch:
                # dwa starts working
                assert all(np.isclose(dwa.weights[k], weights_epoch2[k])
                           for k in weights_epoch2.keys())
                assert_allclose(
                    loss.numpy(),
                    sum(weights_epoch2[k]*losses_epoch[epoch][k][0]
                        for k in weights_epoch2.keys())
                )


@pytest.mark.parametrize('temperature', (1, 2))
@pytest.mark.parametrize('scale', (False, True))
def test_rlw(temperature, scale):
    """Test RandomLossWeighting"""
    # test
    rlw = RandomLossWeighting(
        loss_keys_to_consider=('loss0', 'loss1'),
        temperature=temperature, scale=scale
    )

    # minic sanity check
    loss = rlw.reduce_losses(
        losses={'loss0': torch.tensor(1), 'loss1': torch.tensor(2)},
        batch_idx=0
    )
    assert loss != 0    # it is random, so it is quite hard to test something
    weights = deepcopy(rlw.weights)
    rlw.reset_weights()
    assert weights != rlw.weights   # new random weights -> should be different

    for batch_idx in range(10):
        loss = rlw.reduce_losses(
           losses={'loss0': torch.tensor(1), 'loss1': torch.tensor(2)},
           batch_idx=batch_idx
        )
        if scale:
            # weights always must sum to number of tasks
            assert np.isclose(sum(rlw.weights.values()), 2)
        else:
            assert np.isclose(sum(rlw.weights.values()), 1)

        assert loss != 0    # it is random ...


def test_fixed_weighting():
    """Test FixedLossWeighting"""
    n_batches = 3
    loss_weights = {'loss0': 10, 'loss1': 13}
    losses_epoch = {
        0: {
            'loss0': torch.ones((n_batches,)) * 1,
            'loss1': torch.ones((n_batches,)) * 2,
            'additional_loss': torch.ones((n_batches,)) * 50
        },
        1: {
            'loss0': torch.ones((n_batches,)) * 3,
            'loss1': torch.ones((n_batches,)) * 4,
            'additional_loss': torch.ones((n_batches,)) * 51
        }
    }

    # test
    fixed_weighting_module = FixedLossWeighting(
        weights=loss_weights
    )

    # minic sanity check
    loss = fixed_weighting_module.reduce_losses(
        losses={'loss0': torch.tensor(7), 'loss1': torch.tensor(8)},
        batch_idx=0
    )
    assert loss == 7*10 + 8*13
    fixed_weighting_module.reset_weights()
    assert fixed_weighting_module.weights == loss_weights

    for epoch in range(2):
        for batch_idx in range(n_batches):
            assert fixed_weighting_module.weights == loss_weights

            loss = fixed_weighting_module.reduce_losses(
                losses={k: losses_epoch[epoch][k][batch_idx]
                        for k in losses_epoch[epoch]},
                batch_idx=batch_idx
            )
            if 0 == epoch:
                assert loss == 1*10 + 2*13
            elif 1 == epoch:
                assert loss == 3*10 + 4*13
