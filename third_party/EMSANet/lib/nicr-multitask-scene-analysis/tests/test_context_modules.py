# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import pytest
import torch

from nicr_mt_scene_analysis.model.context_module import get_context_module
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model


def context_module_test(context_module, train_input_size, input_size,
                        upsampling, training, tmp_path):
    cm = get_context_module(
        name=context_module,
        n_channels_in=512,
        n_channels_out=512,
        input_size=train_input_size,
        normalization='batchnorm',
        activation='relu',
        upsampling=upsampling
    )

    if training:
        cm.train()
    else:
        cm.eval()

    x = torch.rand((2, 512, )+input_size)
    _, features_context = cm(x)

    filename = f'context_module_{context_module}'
    filename += f'__up_{upsampling}'
    filename += f'__train_{training}'
    filename += f'__input_{input_size[0]}x{input_size[1]}'
    filename += '.onnx'
    output_names = ['cm_out']
    output_names += [f'features_context_{i}'
                     for i in range(len(features_context))]
    filepath = os.path.join(tmp_path, filename)
    export_onnx_model(filepath, cm, (x, ),
                      input_names=['input'],
                      output_names=output_names)


@pytest.mark.parametrize('context_module', ('ppm', 'appm', 'none'))
@pytest.mark.parametrize('input_size', ((480//32, 640//32),
                                        (960//32, 1280//32)))
@pytest.mark.parametrize('upsampling', ('bilinear', 'nearest'))
@pytest.mark.parametrize('training', (False, True))
def test_context_module1(context_module, input_size, upsampling, training,
                         tmp_path):
    """Test context modules with 640x480 inputs"""
    train_input_size = (480//32, 640//32)
    context_module_test(context_module, train_input_size, input_size,
                        upsampling, training, tmp_path)


@pytest.mark.parametrize('context_module', ('ppm-1-2-4-8', 'appm-1-2-4-8',
                                            'none'))
@pytest.mark.parametrize('input_size', ((512//32, 1024//32),
                                        (1024//32, 2048//32)))
@pytest.mark.parametrize('upsampling', ('bilinear', 'nearest'))
@pytest.mark.parametrize('training', (False, True))
def test_context_module2(context_module, input_size, upsampling, training,
                         tmp_path):
    """Test context modules with 1024x512 inputs (Cityscapes)"""
    train_input_size = (512//32, 1024//32)
    context_module_test(context_module, train_input_size, input_size,
                        upsampling, training, tmp_path)


@pytest.mark.parametrize('context_module', ('ppm-1-5-10', 'appm-1-5-10',
                                            'none'))
@pytest.mark.parametrize('input_size', ((960//32, 1280//32),))
@pytest.mark.parametrize('upsampling', ('bilinear', 'nearest'))
@pytest.mark.parametrize('training', (False, True))
def test_context_module1(context_module, input_size, upsampling, training,
                         tmp_path):
    """Test context modules with 1280x960 inputs"""
    train_input_size = (960//32, 1280//32)
    context_module_test(context_module, train_input_size, input_size,
                        upsampling, training, tmp_path)
