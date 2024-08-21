# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import pytest
import torch

from nicr_mt_scene_analysis.model.encoder_fusion import get_encoder_fusion_class
from nicr_mt_scene_analysis.model.activation import get_activation_class
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model


@pytest.mark.parametrize('fusion', ('se-add', 'add',
                                    'add-uni-rgb', 'add-uni-depth',
                                    'se-add-uni-rgb', 'se-add-uni-depth',
                                    'none'))
@pytest.mark.parametrize('activation', ('relu', 'swish'))
@pytest.mark.parametrize('channels_last', (False, True))
def test_encoder_fusion(fusion, activation, channels_last, tmp_path):
    """Test encoder fusions"""
    # define inputs
    n_channels = 32    # for default reduction in SE > 16
    x_rgb = torch.randn((2, n_channels, 240, 320))
    x_depth = torch.randn((2, n_channels, 240, 320))

    # inputs may also have channels last (NHWC)
    if channels_last:
        x_rgb = x_rgb.permute(0, 2, 3, 1)
        x_depth = x_depth.permute(0, 2, 3, 1)

    # create fusion module
    fusion_class = get_encoder_fusion_class(fusion)
    fusion_module = fusion_class(
        n_channels_in=n_channels,
        activation=get_activation_class(activation),
        input_memory_layout='nchw' if not channels_last else 'nhwc',
    )

    # apply fusion module
    out = fusion_module({'rgb': x_rgb, 'depth': x_depth})
    assert len(out) == 2
    assert out['rgb'].shape == x_rgb.shape
    assert out['depth'].shape == x_depth.shape

    # export model
    filename = f'encoder_fusion_{fusion}__act_{activation}.onnx'
    filepath = os.path.join(tmp_path, filename)
    input_names = ['input_rgb', 'input_depth']
    output_names = ['output_rgb', 'output_depth']
    export_onnx_model(filepath, fusion_module, ((x_rgb, x_depth),),
                      input_names=input_names,
                      output_names=output_names)
