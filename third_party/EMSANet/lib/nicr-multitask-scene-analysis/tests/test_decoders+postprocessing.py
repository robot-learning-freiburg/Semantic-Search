# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmeanu.de>
"""
import os

import pytest
import torch

from nicr_mt_scene_analysis.model.activation import get_activation_class
from nicr_mt_scene_analysis.model.block import get_block_class
from nicr_mt_scene_analysis.model.decoder import SemanticDecoder
from nicr_mt_scene_analysis.model.decoder import SemanticMLPDecoder
from nicr_mt_scene_analysis.model.decoder import InstanceDecoder
from nicr_mt_scene_analysis.model.decoder import InstanceMLPDecoder
from nicr_mt_scene_analysis.model.decoder import NormalDecoder
from nicr_mt_scene_analysis.model.decoder import NormalMLPDecoder
from nicr_mt_scene_analysis.model.decoder import PanopticHelper
from nicr_mt_scene_analysis.model.decoder import SceneClassificationDecoder
from nicr_mt_scene_analysis.model.encoder_decoder_fusion import get_encoder_decoder_fusion_class
from nicr_mt_scene_analysis.model.normalization import get_normalization_class
from nicr_mt_scene_analysis.model.postprocessing import get_postprocessing_class
from nicr_mt_scene_analysis.model.upsampling import get_upsampling_class
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model


def decoders_test(tasks,
                  use_swin_encoder,
                  downsampling_in,
                  use_mlp_decoder,
                  panoptic_enabled,
                  do_postprocessing,
                  training,
                  debug,
                  tmp_path):

    input_h, input_w = (480, 640)

    # set fixed seed (cherry-picked seed)
    # necessary to ensure passing instance tests with random input and weights
    torch.manual_seed(0)

    # determine encoder decoder fusion class
    fusion_name = 'swin-ln-' if use_swin_encoder else ''
    fusion_name += 'select-rgb' if use_mlp_decoder else 'add-rgb'

    # determine parameters depending on decoder type
    if use_mlp_decoder:
        # common parameters used in almost all decoders
        common_kwargs = {
            'n_channels_in': 512,
            'n_channels': (256, 128, 64, 64),
            'downsampling_in': downsampling_in,
            'fusion': get_encoder_decoder_fusion_class(fusion_name),
            'fusion_n_channels': (256, 128, 64),
            'fusion_downsamplings': (16, 8, 4),
            'downsampling_in_heads': 4,
            'normalization': get_normalization_class('batchnorm'),
            'activation': get_activation_class('relu'),
            'upsampling': get_upsampling_class('bilinear'),
            'prediction_upsampling': get_upsampling_class('learned-3x3-zeropad'),
        }
        SemanticDecoderClass = SemanticMLPDecoder
        InstanceDecoderClass = InstanceMLPDecoder
        NormalDecoderClass = NormalMLPDecoder
    else:
        # common parameters used in almost all decoders
        common_kwargs = {
            'n_channels_in': 512,
            'n_channels': (512, 256, 128),
            'downsampling_in': downsampling_in,
            'downsamplings': (16, 8, 4),
            'fusion': get_encoder_decoder_fusion_class(fusion_name),
            'fusion_n_channels': (256, 128, 64),
            'fusion_downsamplings': (16, 8, 4),
            'normalization': get_normalization_class('batchnorm'),
            'activation': get_activation_class('relu'),
            'upsampling': get_upsampling_class('learned-3x3-zeropad'),
            'prediction_upsampling': get_upsampling_class('learned-3x3-zeropad'),
            'block': get_block_class('nonbottleneck1d'),
            'n_blocks': 1,
        }
        SemanticDecoderClass = SemanticDecoder
        InstanceDecoderClass = InstanceDecoder
        NormalDecoderClass = NormalDecoder

    if 'semantic' in tasks:
        semantic_decoder = SemanticDecoderClass(
            n_classes=40,
            postprocessing=get_postprocessing_class('semantic', debug=debug),
            **common_kwargs
        )
        decoder = semantic_decoder
    if 'instance' in tasks:
        instance_decoder = InstanceDecoderClass(
            n_channels_per_task=32,
            with_orientation=('orientation' in tasks),
            postprocessing=get_postprocessing_class(
                'instance',
                heatmap_threshold=0.1,
                heatmap_nms_kernel_size=3,
                top_k_instances=64,
                debug=debug
            ),
            **common_kwargs
        )
        decoder = instance_decoder

    if panoptic_enabled:
        decoder = PanopticHelper(
            semantic_decoder=semantic_decoder,
            instance_decoder=instance_decoder,
            postprocessing=get_postprocessing_class(
                'panoptic',
                semantic_postprocessing=semantic_decoder.postprocessing,
                instance_postprocessing=instance_decoder.postprocessing,
                semantic_classes_is_thing=(True, )*40,
                semantic_class_has_orientation=(True, )*40,
                compute_scores=True,
                debug=debug
            )
        )

    if 'normal' in tasks:
        decoder = NormalDecoderClass(
            n_channels_out=3,
            postprocessing=get_postprocessing_class('normal', debug=debug),
            **common_kwargs
        )

    if 'scene' in tasks:
        common_kwargs['n_channels_in'] = 512//2
        decoder = SceneClassificationDecoder(
            n_classes=10,
            postprocessing=get_postprocessing_class('scene', debug=debug),
            **common_kwargs
        )

    # set up inputs for decoders
    x = (
        # output of context module
        torch.rand(3, 512, input_h//downsampling_in, input_w//downsampling_in),
        # at least one context branch (from GAP)
        (torch.rand(3, 512//2, 1, 1),)
    )
    if use_swin_encoder:
        # NHWC skip connections
        # strings are used to prevent casting keys from int to tensor(int)
        # while exporting to ONNX
        skips = {
            '16': {'rgb': torch.rand(3, input_h//16, input_w//16, 256)},
            '8': {'rgb': torch.rand(3, input_h//8, input_w//8, 128)},
            '4': {'rgb': torch.rand(3, input_h//4, input_w//4, 64)},
        }
    else:
        # NCHW skip connections
        # strings are used to prevent casting keys from int to tensor(int)
        # while exporting to ONNX
        skips = {
            '16': {'rgb': torch.rand(3, 256, input_h//16, input_w//16)},
            '8': {'rgb': torch.rand(3, 128, input_h//8, input_w//8)},
            '4': {'rgb': torch.rand(3, 64, input_h//4, input_w//4)},
        }

    batch = {}
    if 'instance' in tasks:
        # pure instance segmentation task requires gt foreground mask
        batch['instance_foreground'] = torch.ones((3, input_h, input_w),
                                                  dtype=torch.bool)
    if 'orientation' in tasks:
        # orientation estimation requires a gt segmentation and foreground mask
        batch['instance'] = torch.ones((3, 480, 640), dtype=torch.bool)
        batch['orientation_foreground'] = torch.ones((3, input_h, input_w),
                                                     dtype=torch.bool)

    # we need at least one fullres key in the batch for shape deriving in
    # inference (validation) postprocessing
    batch['rgb_fullres'] = torch.ones((3, 3, input_h, input_w),
                                      dtype=torch.uint8)

    if not training:
        decoder.eval()

    output = decoder(x, skips, batch,
                     do_postprocessing=do_postprocessing)

    if not do_postprocessing:
        assert isinstance(output, tuple)
        assert len(output) == 2
    else:
        # postprocessed output of decoder is returned: dict
        assert isinstance(output, dict)
        assert len(output)

        keys = []
        if 'semantic' in tasks or panoptic_enabled:
            keys += ['semantic_output', 'semantic_side_outputs']
            if not training:
                keys += ['semantic_softmax_scores',
                         'semantic_segmentation_score',
                         'semantic_segmentation_idx',
                         'semantic_output_fullres',
                         'semantic_softmax_scores_fullres',
                         'semantic_segmentation_score_fullres',
                         'semantic_segmentation_idx_fullres']

        if 'instance' in tasks or panoptic_enabled:
            keys += ['instance_output', 'instance_side_outputs']
            if not training:
                keys += ['instance_centers',
                         'instance_offsets',
                         'instance_segmentation_gt_foreground',
                         'instance_segmentation_gt_meta',
                         'instance_segmentation_gt_foreground_fullres']
                if debug:
                    keys += ['instance_segmentation_all_foreground',
                             'instance_segmentation_all_foreground_fullres']

        if panoptic_enabled:
            if not training:
                keys += ['panoptic_foreground_mask',
                         'panoptic_segmentation_deeplab',
                         'panoptic_segmentation_deeplab_fullres',
                         'panoptic_segmentation_deeplab_ids',
                         'panoptic_segmentation_deeplab_semantic_idx',
                         'panoptic_segmentation_deeplab_semantic_idx_fullres',
                         'panoptic_segmentation_deeplab_semantic_score',
                         'panoptic_segmentation_deeplab_semantic_score_fullres',
                         'panoptic_segmentation_deeplab_instance_idx',
                         'panoptic_segmentation_deeplab_instance_idx_fullres',
                         'panoptic_segmentation_deeplab_instance_meta',
                         'panoptic_segmentation_deeplab_instance_score',
                         'panoptic_segmentation_deeplab_instance_score_fullres',
                         'panoptic_segmentation_deeplab_panoptic_score',
                         'panoptic_segmentation_deeplab_panoptic_score_fullres']
                if 'orientation' in tasks:
                    keys += ['orientations_panoptic_segmentation_deeplab_instance']

        if 'orientation' in tasks:
            if not training:
                keys += ['orientations_gt_instance_gt_orientation_foreground',
                         'orientations_instance_segmentation_gt_orientation_foreground']
                if debug:
                    keys += ['orientations_gt_instance',
                             'orientations_instance_segmentation']

        if 'normal' in tasks:
            keys += ['normal_output', 'normal_side_outputs']

        if 'scene' in tasks:
            keys += ['scene_output']
            if not training:
                keys += ['scene_class_score', 'scene_class_idx']

        # check that all keys are present
        for k in keys:
            assert k in list(output.keys())

    # export decoders to ONNX
    if not training and do_postprocessing:
        # stop here: inference postprocessing is challenging
        return
    # determine filename and filepath
    tasks_str = '+'.join(tasks)
    if panoptic_enabled:
        tasks_str += '+panoptic'
    filename = f'decoders_{tasks_str}'
    filename += f'__downsampling_{downsampling_in}'
    filename += f'__mlp_{use_mlp_decoder}'
    filename += f'__swin_{use_swin_encoder}'
    filename += f'__train_{training}'
    filename += f'__post_{do_postprocessing}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    # export
    x = (x, skips, batch, {'do_postprocessing': do_postprocessing})
    export_onnx_model(filepath, decoder, x)


@pytest.mark.parametrize('task', ('semantic',
                                  'instance',
                                  ('instance', 'orientation'),
                                  'normal',
                                  'scene'))
@pytest.mark.parametrize('use_swin_encoder', (False, True))
@pytest.mark.parametrize('downsampling_in', (32, 16))
@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
@pytest.mark.parametrize('debug', (False, True))
def test_decoders_single_decoder(task, use_swin_encoder, downsampling_in,
                                 do_postprocessing, training, debug, tmp_path):
    """Test in single EMSANet decoder setup"""
    tasks = task if isinstance(task, tuple) else (task,)
    decoders_test(tasks=tasks,
                  use_swin_encoder=use_swin_encoder,
                  downsampling_in=downsampling_in,
                  use_mlp_decoder=False,
                  panoptic_enabled=False,
                  do_postprocessing=do_postprocessing,
                  training=training,
                  debug=debug,
                  tmp_path=tmp_path)


@pytest.mark.parametrize('task', ('semantic',
                                  'instance',
                                  ('instance', 'orientation'),
                                  'normal'))
@pytest.mark.parametrize('use_swin_encoder', (False, True))
@pytest.mark.parametrize('downsampling_in', (32, 16))
@pytest.mark.parametrize('training', (False, True))
@pytest.mark.parametrize('debug', (False, True))
def test_decoders_single_mlp_decoder(task, use_swin_encoder, downsampling_in,
                                     training, debug, tmp_path):
    """Test in single MLP decoder setup"""
    tasks = task if isinstance(task, tuple) else (task,)
    decoders_test(tasks=tasks,
                  use_swin_encoder=use_swin_encoder,
                  downsampling_in=downsampling_in,
                  use_mlp_decoder=True,
                  panoptic_enabled=False,
                  do_postprocessing=False,   # tested above
                  training=training,
                  debug=debug,
                  tmp_path=tmp_path)


@pytest.mark.parametrize('tasks', (('semantic', 'instance'),
                                   ('semantic', 'instance', 'orientation')))
@pytest.mark.parametrize('downsampling_in', (32, 16))
@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
@pytest.mark.parametrize('debug', (False, True))
def test_decoders_panoptic(tasks, downsampling_in, do_postprocessing,
                           training, debug, tmp_path):
    """Test in panoptic segmentation setting"""
    decoders_test(tasks=tasks,
                  use_swin_encoder=False,
                  downsampling_in=downsampling_in,
                  use_mlp_decoder=False,
                  panoptic_enabled=True,
                  do_postprocessing=do_postprocessing,
                  training=training,
                  debug=debug,
                  tmp_path=tmp_path)
