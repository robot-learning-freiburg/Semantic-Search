# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from copy import deepcopy
from mimetypes import suffix_map
from sys import prefix

import cv2
import numpy as np
import pytest
from skimage import data
import torch
import torchvision
from torchvision.transforms import InterpolationMode

from nicr_mt_scene_analysis.data.preprocessing.rgb import adjust_hsv
from nicr_mt_scene_analysis.data.preprocessing import CloneEntries
from nicr_mt_scene_analysis.data.preprocessing import FlatCloneEntries
from nicr_mt_scene_analysis.data.preprocessing import InstanceClearStuffIDs
from nicr_mt_scene_analysis.data.preprocessing import InstanceTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import MultiscaleSupervisionGenerator
from nicr_mt_scene_analysis.data.preprocessing import NormalizeRGB
from nicr_mt_scene_analysis.data.preprocessing import NormalizeDepth
from nicr_mt_scene_analysis.data.preprocessing import OrientationTargetGenerator
from nicr_mt_scene_analysis.data.preprocessing import RandomCrop
from nicr_mt_scene_analysis.data.preprocessing import RandomHorizontalFlip
from nicr_mt_scene_analysis.data.preprocessing import RandomHSVJitter
from nicr_mt_scene_analysis.data.preprocessing import RandomResize
from nicr_mt_scene_analysis.data.preprocessing import Resize
from nicr_mt_scene_analysis.data.preprocessing import SemanticClassMapper
from nicr_mt_scene_analysis.data.preprocessing import ToTorchTensors
from nicr_mt_scene_analysis.data.preprocessing import TorchTransformWrapper
from nicr_mt_scene_analysis.data.preprocessing.utils import _get_relevant_spatial_keys
from nicr_mt_scene_analysis.testing.dataset import get_dataset
from nicr_mt_scene_analysis.testing.preprocessing import get_dummy_sample
from nicr_mt_scene_analysis.testing.preprocessing import show_results
from nicr_mt_scene_analysis.utils import np_biternion2rad


@pytest.mark.parametrize('keys_to_clone', (None, ('rgb', 'depth')))
def test_cloneentries(keys_to_clone):
    """Test CloneEntries"""
    pre = CloneEntries(keys_to_clone=keys_to_clone)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    if keys_to_clone is None:
        keys_to_clone = tuple(sample.keys())

    assert len(sample_pre) == len(sample) + 1
    assert len(sample_pre[pre.clone_key]) == len(keys_to_clone)

    clone = sample_pre[pre.clone_key]
    for key in keys_to_clone:
        assert key in clone

        entry = sample[key]
        entry_clone = clone[key]

        if isinstance(entry, np.ndarray):
            assert id(entry_clone.data) != id(entry.data)
            assert np.equal(entry_clone, entry).all()
        elif isinstance(entry, (int, str)):
            # yeah it is python, we are not able to check this
            pass
        else:
            assert id(entry_clone) != id(entry)
            assert entry_clone == entry


@pytest.mark.parametrize('keys_to_clone', (None, ('rgb', 'depth')))
@pytest.mark.parametrize('prefix_and_suffix', ((None, '_xyz'), ('abc_', None),
                                               ('abc_', '_xyz')))
def test_flatcloneentries(keys_to_clone, prefix_and_suffix):
    """Test FlatCloneEntries"""
    prefix, suffix = prefix_and_suffix

    pre = FlatCloneEntries(
        keys_to_clone=keys_to_clone,
        key_prefix=prefix,
        key_suffix=suffix
    )

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    if keys_to_clone is None:
        keys_to_clone = tuple(sample.keys())

    assert len(sample_pre) == len(sample) + len(keys_to_clone)

    for key in keys_to_clone:
        key_clone = (prefix or '') + key + (suffix or '')

        assert key_clone in sample_pre

        entry = sample[key]
        entry_clone = sample_pre[key_clone]

        if isinstance(entry, np.ndarray):
            assert id(entry_clone.data) != id(entry.data)
            assert np.equal(entry_clone, entry).all()
        elif isinstance(entry, (int, str)):
            # yeah it is python, we are not able to check this
            pass
        else:
            assert id(entry_clone) != id(entry)
            assert entry_clone == entry


def test_instanceclearstuffids():
    """Test InstanceClearStuffIDs"""
    thing_classes = [1, 2, 3]
    classes_is_thing = [True if i in thing_classes else False
                        for i in range(41)]
    pre = InstanceClearStuffIDs(
        semantic_classes_is_thing=classes_is_thing
    )
    # apply preprocessor
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # verify that all stuff pixels have instance id 0
    for c in classes_is_thing[1:]:    # skip void
        if c:
            continue
        # stuff class
        assert (sample_pre['instance'][sample_pre['semantic'] == c] == 0).all()

    # some simple checks
    assert 'instance' in sample_pre

    show_results(sample, sample_pre, f'InstanceClearStuffIDs')


def test_instancetargetgenerator():
    """Test InstanceTargetGenerator"""
    relevant_semantic_classes = [1, 2, 3]    # see get_dummy_sample
    classes = [True if i in relevant_semantic_classes else False
               for i in range(41)]
    pre = InstanceTargetGenerator(
        sigma=8,
        semantic_classes_is_thing=classes
    )

    # get sample (and clear instance ids for stuff classes -> get_dummy_sample)
    sample = get_dummy_sample()
    sample = InstanceClearStuffIDs(semantic_classes_is_thing=classes)(sample)

    # apply preprocessor
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert 'instance_center' in sample_pre
    assert 'instance_offset' in sample_pre
    assert 'instance_foreground' in sample_pre

    # check foreground mask
    assert np.all(
        np.equal(sample_pre['instance_foreground'],
                 np.isin(sample['semantic'], relevant_semantic_classes))
    )
    # more checks are quite complicated, you should consider visual inspection
    # first

    show_results(sample, sample_pre, f'InstanceTargetGenerator')


@pytest.mark.parametrize('downscales', ((2,), (4, 8), (8, 16, 32)))
def test_multiscale(downscales):
    """Test MultiscaleSupervisionGenerator"""
    pre = MultiscaleSupervisionGenerator(
        downscales=downscales,
        keys=('instance', 'semantic', 'orientations')
    )

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    for ds in downscales:
        ms_key = f'_down_{ds}'

        assert ms_key in sample_pre

        ms_sample = sample_pre[ms_key]
        for key in ms_sample:
            assert key in sample

        for key in _get_relevant_spatial_keys(ms_sample):
            assert ms_sample[key].shape[0] == int(sample[key].shape[0] / ds)

    show_results(sample, sample_pre,
                 f'MultiscaleSupervisionGenerator: scales: {downscales}')


def test_orientationtargetgenerator():
    """Test OrientationTargetGenerator"""
    relevant_semantic_classes = [1, 2, 3]
    classes = [True if i in relevant_semantic_classes else False
               for i in range(40)]
    pre = OrientationTargetGenerator(
        semantic_classes_estimate_orientation=classes
    )

    # get dummy sample
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))
    valid_instance_ids = [1, 2]     # see _get_dummy_sample

    # some simple checks
    assert 'orientation' in sample_pre
    assert 'orientation_foreground' in sample_pre
    assert 'orientations_present' in sample_pre
    o = sample_pre['orientation']
    o_f = sample_pre['orientation_foreground']
    i = sample_pre['instance']
    assert np.equal(np.unique(i[o_f]), valid_instance_ids).all()

    for id_ in valid_instance_ids:
        o_id_rad = np_biternion2rad(o[i == id_])
        deg_rad = sample['orientations'][id_]
        assert (o_id_rad == deg_rad).all()

    show_results(sample, sample_pre,
                 f'OrientationTargetGenerator')


def test_normalize_rgb():
    """Test NormalizeRGB"""
    dtype = 'float32'
    rgb_mean = np.array((0.485, 0.456, 0.406), dtype=dtype) * 255
    rgb_std = np.array((0.229, 0.224, 0.225), dtype=dtype) * 255

    pre = NormalizeRGB(output_dtype=dtype)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].dtype == dtype

    show_results(sample, sample_pre,
                 f'NormalizeRGB')

    # check rgb
    for c_idx in range(3):
        c = sample['rgb'][..., c_idx]
        c_pre = sample_pre['rgb'][..., c_idx]
        c_ref = (c.astype(dtype) - rgb_mean[c_idx]) / rgb_std[c_idx]
        assert np.equal(c_pre, c_ref).all()


@pytest.mark.parametrize('raw_depth', (False, True))
def test_normalize_depth(raw_depth):
    """Test NormalizeDepth"""
    dtype = 'float32'
    depth_mean = np.array(2022.2202, dtype=dtype)
    depth_std = np.array(123.321, dtype=dtype)

    pre = NormalizeDepth(
        depth_mean=depth_mean,
        depth_std=depth_std,
        raw_depth=raw_depth,
        output_dtype=dtype
    )

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['depth'].dtype == dtype

    show_results(sample, sample_pre,
                 f'NormalizeDepth: raw_depth: {raw_depth}, {dtype}')

    # check depth
    depth = sample['depth']
    depth_pre = sample_pre['depth']
    mask = depth != 0 if raw_depth else slice(None)
    depth_ref = depth.astype(dtype)
    depth_ref[mask] = (depth_ref[mask] - depth_mean) / depth_std
    assert np.equal(depth_pre, depth_ref).all()


@pytest.mark.parametrize('height', (100, 200, 1000))
@pytest.mark.parametrize('width', (100, 200, 1000))
def test_resize(height, width):
    """Test Resize"""
    pre = Resize(height=height, width=width)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    assert sample_pre['depth'].dtype == sample['depth'].dtype
    assert sample_pre['some_mask'].dtype == sample['some_mask'].dtype
    assert sample_pre['rgb'].shape == (height, width, 3)
    assert sample_pre['depth'].shape == (height, width)
    assert sample_pre['some_mask'].shape == (height, width)
    # check interpolation
    assert np.equal(sample_pre['rgb'],
                    cv2.resize(sample['rgb'], (width, height),
                               interpolation=cv2.INTER_LINEAR)).all()
    assert np.equal(sample_pre['depth'],
                    cv2.resize(sample['depth'], (width, height),
                               interpolation=cv2.INTER_NEAREST)).all()
    assert np.equal(sample_pre['some_mask'],
                    cv2.resize(sample['some_mask'].astype('uint8'),
                               (width, height),
                               interpolation=cv2.INTER_NEAREST) > 0).all()

    show_results(sample, sample_pre, f'Resize: h: {height}, w: {width}')


@pytest.mark.parametrize('min_scale', (0.8, 1.0))
@pytest.mark.parametrize('max_scale', (1.0, 1.4))
def test_randomresize(min_scale, max_scale):
    """Test RandomResize"""
    pre = RandomResize(min_scale=min_scale,
                       max_scale=max_scale)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    # note that results are checked in test_resize
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    assert sample_pre['depth'].dtype == sample['depth'].dtype
    assert sample_pre['some_mask'].dtype == sample['some_mask'].dtype
    shape = sample_pre['rgb'].shape[:2]
    assert sample_pre['depth'].shape == sample_pre['some_mask'].shape == shape

    show_results(sample, sample_pre,
                 f'RandomResize: min: {min_scale}, max: {max_scale}')


@pytest.mark.parametrize('crop_height', (100, 200, 1000))
@pytest.mark.parametrize('crop_width', (100, 200, 1000))
def test_randomcrop(crop_height, crop_width):
    """Test RandomCrop"""
    pre = RandomCrop(crop_height=crop_height,
                     crop_width=crop_width)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    shape = sample_pre['rgb'].shape[:2]
    assert sample_pre['depth'].shape == sample_pre['some_mask'].shape == shape

    show_results(sample, sample_pre,
                 f'RandomCrop: h: {crop_height}, w: {crop_width}')


@pytest.mark.parametrize('p', (0.1, 0.5, 0.8, 1.0))
def test_randomhorizontalflip(p):
    """Test RandomHorizontalFlip"""
    pre = RandomHorizontalFlip(p=p)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    if 1.0 == p:
        for k in _get_relevant_spatial_keys(sample):
            assert np.equal(sample_pre[k], sample[k][:, ::-1]).all()
        for id_ in sample['orientations']:
            angle = np.rad2deg(sample['orientations'][id_])
            angle_ref = (360 - angle) % 360

            assert np.deg2rad(angle_ref) == sample_pre['orientations'][id_]

    show_results(sample, sample_pre, f'RandomFlip: p: {p}')


@pytest.mark.parametrize('hue_jitter', (10/(360./2),))    # +-10 degrees
@pytest.mark.parametrize('saturation_jitter', (20/(255.),))    # +- ~8%
@pytest.mark.parametrize('value_jitter', (50/(255.),))    # +- ~16%
def test_hsvjitter(hue_jitter, saturation_jitter, value_jitter):
    """Test RandomHSVJitter"""
    pre = RandomHSVJitter(hue_jitter=hue_jitter,
                          saturation_jitter=saturation_jitter,
                          value_jitter=value_jitter)

    # apply augmentation
    sample = get_dummy_sample()
    sample_pre = pre(deepcopy(sample))

    # some simple checks
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    shape = sample_pre['rgb'].shape[:2]
    assert sample_pre['depth'].shape == sample_pre['some_mask'].shape == shape

    show_results(sample, sample_pre,
                 f'RandomHSVJitter: h: {hue_jitter*180}, '
                 f's: {saturation_jitter*255}, v: {value_jitter*255}')


@pytest.mark.parametrize('img', (data.astronaut(), data.chelsea(), data.coffee()))
@pytest.mark.parametrize('hue_jitter_offset', (10//2, -10//2))
@pytest.mark.parametrize('saturation_jitter_offset', (20, -20))
@pytest.mark.parametrize('value_jitter_offset', (50, -50))
def test_hsvjitter_limits(img,
                          hue_jitter_offset,
                          saturation_jitter_offset,
                          value_jitter_offset):
    """Test to visualize limits of HSV color jittering"""
    # adjust color in hsv space
    img_pre = adjust_hsv(img,
                         hue_jitter_offset,
                         saturation_jitter_offset,
                         value_jitter_offset)

    show_results({'rgb': img}, {'rgb': img_pre},
                 f'HSVJitterLimits: h: {hue_jitter_offset*2}, '
                 f's: {saturation_jitter_offset}, v: {value_jitter_offset}')


def test_semanticclassmapper():
    """Test SemanticClassMapper"""
    classes_to_map = (1, 10)    # see get_dummy_samples
    new_label = 0
    pre = SemanticClassMapper(
        classes_to_map=classes_to_map,
        new_label=new_label
    )

    # get sample
    sample = get_dummy_sample()

    # apply preprocessor
    sample_pre = pre(deepcopy(sample))

    # check that mapping was applied
    mask_changed = np.zeros_like(sample['semantic'], dtype=np.bool)
    for c in classes_to_map:
        mask = sample['semantic'] == c
        assert (sample_pre['semantic'][mask] == new_label).all()

        mask_changed = np.logical_or(mask_changed, mask)

    # check that remaining pixels are unchanged
    mask = np.logical_not(mask_changed)
    assert (sample_pre['semantic'][mask] == sample['semantic'][mask]).all()

    show_results(sample, sample_pre, f'SemanticClassMapper')


def test_totorchtensors():
    """Test ToTorchTensors"""
    pre = ToTorchTensors()

    # apply augmentation
    sample = get_dummy_sample()

    # pytorch does not support uint16 as used for depth, we circumvent this by
    # manually casting the depth image to float, in chained processing, this
    # is usually done by NormalizeRGBAndDepth before
    sample['depth'] = sample['depth'].astype('float32')

    sample_pre = pre(deepcopy(sample))

    # some simple checks
    for k, v in sample_pre.items():
        if isinstance(v, (dict, str, int)):
            # primitive types should be unchanged
            assert sample[k] == v
            continue

        assert isinstance(v, torch.Tensor)


@pytest.mark.parametrize('interpolation', (InterpolationMode.NEAREST,
                                           InterpolationMode.BILINEAR))
def test_torchtransformwrapper(interpolation):
    """Test TorchTransformWrapper"""
    if interpolation == InterpolationMode.BILINEAR:
        pytest.xfail('should not work for bilinear interpolation')
    pre = torchvision.transforms.Compose([
        TorchTransformWrapper(torchvision.transforms.RandomResizedCrop(
            10, interpolation=interpolation)),
        TorchTransformWrapper(torchvision.transforms.TenCrop(5))
    ])

    rgb = torch.arange(0, 256, dtype=torch.float64).reshape(1, 16, 16).repeat(3, 1, 1)
    depth = rgb[0, ...].unsqueeze(0).type(torch.float32) * 100

    sample = {'rgb': rgb, 'depth': depth}

    sample_pre = pre(deepcopy(sample))

    assert sample_pre['rgb'].shape == (10, 3, 5, 5)
    assert sample_pre['depth'].shape == (10, 1, 5, 5)
    assert sample_pre['rgb'].dtype == sample['rgb'].dtype
    assert sample_pre['depth'].dtype == sample['depth'].dtype
    assert torch.allclose(
        sample_pre['rgb'][:, 0, :, :].unsqueeze(1),
        sample_pre['depth'].type(sample_pre['rgb'].dtype) / 100
    )


def test_qualitative_torchtransform_wrapper():
    # second just qualitative test with real images
    class ToFloat:
        def __call__(self, sample):
            for key in _get_relevant_spatial_keys(sample):
                sample[key] = sample[key].astype(np.float32)
            return sample

    pre = torchvision.transforms.Compose([
        ToFloat(),
        ToTorchTensors(),
        TorchTransformWrapper(torchvision.transforms.RandomResizedCrop(
            300, interpolation=InterpolationMode.NEAREST)),
        TorchTransformWrapper(torchvision.transforms.TenCrop(250))
    ])

    dataset = get_dataset(
        name='nyuv2',
        split='train',
        sample_keys=('rgb', 'depth')
        )

    for i in range(5):
        sample = dataset[i]
        sample_pre = pre(deepcopy(sample))

        for crop_rgb, crop_depth in zip(sample_pre['rgb'], sample_pre['depth']):
            crop_rgb = crop_rgb.numpy().transpose(1, 2, 0).astype(np.uint8)
            crop_depth = crop_depth.numpy().transpose(1, 2, 0).astype(np.uint16)
            show_results(
                sample,
                {'rgb': crop_rgb, 'depth': crop_depth},
                'TorchTransformWrapper with rgb and depth')
