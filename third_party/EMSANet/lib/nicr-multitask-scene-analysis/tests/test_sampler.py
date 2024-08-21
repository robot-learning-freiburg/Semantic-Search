# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import pytest
from torch.utils.data import DataLoader

from nicr_mt_scene_analysis.data import RandomSamplerSubset


@pytest.mark.parametrize('subset', (1.0, 0.5))
@pytest.mark.parametrize('deterministic', (True, False))
def test_sampler(deterministic, subset):
    """Test Random Subsampler"""
    dataset = list(range(500))
    sampler = RandomSamplerSubset(
        data_source=dataset,
        subset=subset,
        deterministic=deterministic
    )

    # iteration 1 (first epoch)
    a = list(sampler)

    # iteration 2 (another epoch)
    b = list(sampler)

    # check if subset is properly applied
    assert len(a) == int(len(dataset) * subset)

    # data should be shuffled, thus in a different order
    assert a != b

    a.sort()
    b.sort()

    if deterministic or subset == 1:
        # after sorting both should be equal for determinist case
        # if subset == 1, then all indices should be present, thus, both should
        # be equal even if deterministic is False
        assert a == b
    else:
        # if subset is set and deterministic is False, we expect to get
        # different indices for each call
        assert a != b


@pytest.mark.parametrize('n_workers', (0, 4, 8))
@pytest.mark.parametrize('persistent_workers', (True, False))
@pytest.mark.parametrize('subset', (1.0, 0.5))
@pytest.mark.parametrize('deterministic', (True, False))
def test_sampler_with_dataloader(n_workers, persistent_workers,
                                 deterministic, subset):
    """Test Random Subsampler together with PyTorch dataloader"""
    batch_size = 10
    dataset = list(range(500))
    sampler = RandomSamplerSubset(
        data_source=dataset,
        subset=subset,
        deterministic=deterministic
    )

    # persistent_workers=True requires num_workers > 0
    persistent_workers = persistent_workers & n_workers > 0

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            drop_last=False,
                            num_workers=n_workers,
                            persistent_workers=persistent_workers)

    # iteration 1 (first epoch)
    a = []
    for batch in dataloader:
        # ensure that multiprocessing does not break sampling
        assert len(set(batch)) == batch_size
        a.extend(batch)

    # iteration 2 (another epoch)
    b = []
    for batch in dataloader:
        # ensure that multiprocessing does not break sampling
        assert len(set(batch)) == batch_size
        b.extend(batch)

    # check if subset is properly applied
    assert len(a) == int(len(dataset) * subset)

    # data should be shuffled, thus in a different order
    assert a != b

    a.sort()
    b.sort()

    if deterministic or subset == 1:
        # after sorting both should be equal for determinist case
        # if subset == 1, then all indices should be present, thus, both should
        # be equal even if deterministic is False
        assert a == b
    else:
        # if subset is set and deterministic is False, we expect to get
        # different indices for each call
        assert a != b
