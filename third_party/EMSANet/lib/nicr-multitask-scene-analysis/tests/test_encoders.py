# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from itertools import product
import os

import pytest
import torch

from nicr_mt_scene_analysis.model.backbone import get_backbone
from nicr_mt_scene_analysis.model.backbone import IS_SWIN_AVAILABLE
from nicr_mt_scene_analysis.model.encoder import get_encoder
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model
from nicr_mt_scene_analysis.testing import EXPORT_ONNX_MODELS


def encoder_test1(modalities,
                  backbone_rgb,
                  backbone_depth,
                  activation,
                  fusion,
                  normalization,
                  tmp_path):
    # build encoder model
    if 'resnet50' in backbone_rgb:
        backbone_rgb_block = 'bottleneck'
    else:
        backbone_rgb_block = 'nonbottleneck1d'

    if 'resnet50' in backbone_depth:
        backbone_depth_block = 'bottleneck'
    else:
        backbone_depth_block = 'nonbottleneck1d'

    decoder_skip_downsamplings = (4, 8, 16)

    encoder_input = {}

    if 'rgb' in modalities:
        x_rgb = torch.randn((2, 3, 480, 640))
        encoder_input['rgb'] = x_rgb

        backbone_rgb = get_backbone(
            name=backbone_rgb,
            resnet_block=backbone_rgb_block,
            n_input_channels=3,
            normalization=normalization,
            activation=activation,
            pretrained=False,
        )
    else:
        backbone_rgb = None

    if 'depth' in modalities:
        x_depth = torch.randn((2, 1, 480, 640))
        encoder_input['depth'] = x_depth

        backbone_depth = get_backbone(
            name=backbone_depth,
            resnet_block=backbone_depth_block,
            n_input_channels=1,
            normalization=normalization,
            activation=activation,
            pretrained=False,
        )
    else:
        backbone_depth = None

    # create encoder
    encoder = get_encoder(
        backbone_rgb=backbone_rgb,
        backbone_depth=backbone_depth,
        backbone_rgbd=None,
        fusion=fusion,
        normalization=normalization,
        activation=activation,
        skip_downsamplings=decoder_skip_downsamplings
    )

    # apply model
    encoder_output = encoder(encoder_input)

    # some simple checks
    keys = sorted(encoder_input.keys())
    # -> we have a tuple (output, skips), both elements are tuples
    assert len(encoder_output) == 2
    assert all(isinstance(eo, dict) for eo in encoder_output)
    # -> all keys in output must still be present
    assert len(encoder_output[0]) == len(encoder_input)
    assert sorted(encoder_output[0].keys()) == keys
    # -> skip connections must be a dict of dicts that as well contain all keys
    assert len(encoder_output[1]) == len(decoder_skip_downsamplings)
    assert all(sorted(encoder_output[1][str(ds)].keys()) == keys
               for ds in decoder_skip_downsamplings)

    # export model
    modalities_str = '+'.join(modalities)
    filename = f'encoder_{modalities_str}'
    filename += f'__backbone_rgb_{backbone_rgb}'
    filename += f'__backbone_depth_{backbone_depth}'
    filename += f'__act_{activation}'
    filename += f'__fusion_{fusion}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    input_names = [f'input_{m}' for m in modalities]
    output_names = [f'output_{m}' for m in modalities]
    output_names += [f'skip_{ds}_{m}'
                     for ds, m in product(decoder_skip_downsamplings,
                                          modalities)]
    export_onnx_model(filepath, encoder, (encoder_input, {}),
                      input_names=input_names,
                      output_names=output_names)


def encoder_test2(modalities,
                  backbone_rgbd,
                  normalization,
                  tmp_path):
    # build encoder model

    decoder_skip_downsamplings = (4, 8, 16)

    encoder_input = {}

    x_rgbd = torch.randn((2, 4, 480, 640))
    encoder_input['rgbd'] = x_rgbd

    backbone_rgbd = get_backbone(
        name=backbone_rgbd,
        resnet_block=None,
        n_input_channels=4,
        normalization=normalization,
        activation='relu',
        pretrained=False,
    )

    # create encoder
    encoder = get_encoder(
        backbone_rgb=None,
        backbone_depth=None,
        backbone_rgbd=backbone_rgbd,
        normalization=normalization,
        activation='relu',
        skip_downsamplings=decoder_skip_downsamplings
    )

    # apply model
    encoder_output = encoder(encoder_input)

    # some simple checks
    keys = sorted(encoder_input.keys())
    # -> we have a tuple (output, skips), both elements are tuples
    assert len(encoder_output) == 2
    assert all(isinstance(eo, dict) for eo in encoder_output)
    # -> all keys in output must still be present
    assert len(encoder_output[0]) == len(encoder_input)
    assert sorted(encoder_output[0].keys()) == keys
    # -> skip connections must be a dict of dicts that as well contain all keys
    assert len(encoder_output[1]) == len(decoder_skip_downsamplings)
    assert all(sorted(encoder_output[1][str(ds)].keys()) == keys
               for ds in decoder_skip_downsamplings)

    # export model
    modalities_str = '+'.join(modalities)
    filename = f'encoder_{modalities_str}'
    filename += f'__backbone_rgbd_{backbone_rgbd}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    input_names = [f'input_{m}' for m in modalities]
    output_names = [f'output_{m}' for m in modalities]
    output_names += [f'skip_{ds}_{m}'
                     for ds, m in product(decoder_skip_downsamplings,
                                          modalities)]
    export_onnx_model(filepath, encoder, (encoder_input, {}),
                      input_names=input_names,
                      output_names=output_names)


@pytest.mark.xfail(EXPORT_ONNX_MODELS, reason='ONNX export is not supported yet')
@pytest.mark.parametrize('modalities', (('rgb',), ('depth',), ('rgb', 'depth')))
@pytest.mark.parametrize('backbone', ('resnet18', 'resnet50', 'resnet34se'))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder1(modalities, backbone, activation, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with same backbone"""
    encoder_test1(modalities=modalities,
                  backbone_rgb=backbone,
                  backbone_depth=backbone,
                  activation=activation,
                  fusion=fusion,
                  normalization='batchnorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(EXPORT_ONNX_MODELS, reason='ONNX export is not supported yet')
@pytest.mark.parametrize('backbones', (('resnet18', 'resnet34'),))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder2(backbones, activation, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with different backbone"""
    encoder_test1(modalities=('rgb', 'depth'),
                  backbone_rgb=backbones[0],
                  backbone_depth=backbones[1],
                  activation=activation,
                  fusion=fusion,
                  normalization='batchnorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(EXPORT_ONNX_MODELS, reason='ONNX export is not supported yet')
@pytest.mark.xfail(reason='ONNX export is not supported yet')
@pytest.mark.parametrize('backbones', (('resnet18', 'resnet50'),))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder3(backbones, activation, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with different backbone"""
    encoder_test1(modalities=('rgb', 'depth'),
                  backbone_rgb=backbones[0],
                  backbone_depth=backbones[1],
                  activation=activation,
                  fusion=fusion,
                  normalization='batchnorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(EXPORT_ONNX_MODELS, reason='ONNX export is not supported yet')
@pytest.mark.parametrize('modalities', (('rgb',), ('depth',)))
@pytest.mark.parametrize('backbone', ('resnet18', 'resnet34'))
def test_encoder_rgb_or_depth_backbone(modalities, backbone, tmp_path):
    """Test Encoder with single modality backbone"""
    encoder_test1(modalities=modalities,
                  backbone_rgb=backbone,
                  backbone_depth=backbone,
                  activation='relu',
                  fusion=None,
                  normalization='batchnorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(not IS_SWIN_AVAILABLE,
                   reason="Torchvision version is too old")
@pytest.mark.xfail(EXPORT_ONNX_MODELS, reason='ONNX export is not supported yet')
@pytest.mark.parametrize('modalities', (('rgb',), ('depth',), ('rgb', 'depth')))
@pytest.mark.parametrize('backbone', ('swin-t', 'swin-t-v2'))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder_swin(modalities, backbone, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with same backbone"""
    encoder_test1(modalities=modalities,
                  backbone_rgb=backbone,
                  backbone_depth=backbone,
                  activation='relu',
                  fusion=fusion,
                  normalization='layernorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(not IS_SWIN_AVAILABLE,
                   reason="Torchvision version is too old")
@pytest.mark.xfail(EXPORT_ONNX_MODELS, reason='ONNX export is not supported yet')
@pytest.mark.parametrize('backbones', (('swin-t', 'swin-t-v2'),))
@pytest.mark.parametrize('fusion', ('add-uni-rgb', 'se-add-uni-rgb'))
def test_fused_encoder2_swin(backbones, fusion, tmp_path):
    """Test Encoder/FusedRGBDEncoder with different backbone"""
    encoder_test1(modalities=('rgb', 'depth'),
                  backbone_rgb=backbones[0],
                  backbone_depth=backbones[1],
                  activation='relu',
                  fusion=fusion,
                  normalization='layernorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(not IS_SWIN_AVAILABLE,
                   reason="Torchvision version is too old")
@pytest.mark.xfail(EXPORT_ONNX_MODELS, reason='ONNX export is not supported yet')
@pytest.mark.parametrize('modalities', (('rgb',), ('depth',)))
@pytest.mark.parametrize('backbone', ('swin-t', 'swin-t-v2'))
def test_encoder_rgb_or_depth_backbone_swin(modalities, backbone, tmp_path):
    """Test Encoder with single modality backbone"""
    encoder_test1(modalities=modalities,
                  backbone_rgb=backbone,
                  backbone_depth=backbone,
                  activation='relu',
                  fusion=None,
                  normalization='layernorm',
                  tmp_path=tmp_path)


@pytest.mark.xfail(not IS_SWIN_AVAILABLE,
                   reason="Torchvision version is too old")
@pytest.mark.xfail(EXPORT_ONNX_MODELS, reason='ONNX export is not supported yet')
@pytest.mark.parametrize('backbone', ('swin-multi-t',
                                      'swin-multi-t-v2',
                                      'swin-multi-t-128',
                                      'swin-multi-t-v2-128'))
def test_encoder_multimodal_swin(backbone, tmp_path):
    """Test Encoder with single modality backbone"""
    encoder_test2(modalities=('rgbd'),
                  backbone_rgbd=backbone,
                  normalization='layernorm',
                  tmp_path=tmp_path)
