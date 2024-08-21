# -*- coding: utf-8 -*-
"""
.. codeauthor:: Robin Schmidt <robin.schmidt@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

MLP decoder is inspired by:
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.
    https://arxiv.org/abs/2105.15203

"""
import abc
from collections import OrderedDict
from typing import Tuple, Type

import torch
import torch.nn as nn

from ...types import DecoderInputType
from ...types import DecoderRawOutputType
from ...types import EncoderSkipsType
from ..activation import get_activation_class
from ..encoder_decoder_fusion import EncoderDecoderFusionType
from ..normalization import get_normalization_class
from ..postprocessing import PostProcessingType
from ..upsampling import get_upsampling_class
from ..upsampling import UpsamplingType
from ..utils import ConvNormAct
from .base import DecoderBase


class MLPDecoderBase(DecoderBase):
    def __init__(
        self,
        n_channels_in: int,
        downsampling_in: int,
        n_channels: Tuple[int, ...],
        fusion: Type[EncoderDecoderFusionType],
        fusion_n_channels: Tuple[int, ...],
        fusion_downsamplings: Tuple[int, ...],
        postprocessing: Type[PostProcessingType],
        downsampling_in_heads: int = 4,
        dropout_p: float = 0.1,
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: Type[UpsamplingType] = get_upsampling_class()
    ) -> None:
        super().__init__(postprocessing=postprocessing)

        # perform some simple sanity checks
        assert len(n_channels) == 1 + len(fusion_n_channels)
        assert len(fusion_n_channels) == len(fusion_downsamplings)
        assert sorted(fusion_downsamplings,
                      reverse=True) == list(fusion_downsamplings)

        self._fusion_downsamplings = fusion_downsamplings

        # main branch from encoder / context module
        self.main_branch = nn.Sequential(
            OrderedDict([
                ('embedding', ConvNormAct(
                    n_channels_in=n_channels_in,
                    n_channels_out=n_channels[0],
                    kernel_size=1,
                    normalization=None,
                    activation=None
                )),
                ('upsample', upsampling(
                    n_channels=n_channels[0],
                    scale_factor=downsampling_in // downsampling_in_heads
                ))
            ])
        )

        # skip connection branches from encoder
        skip_fusions = []
        for n_skip in fusion_n_channels:
            # encoder-decoder fusion, should be:
            # - ResNet encoder: 'select' / 'select-*'
            # - Swin encoder: 'swin-ln-select' / 'swin-ln-select-*'
            # note, the fusion handles also the layernorm for swin encoders
            skip_fusions.append(
                fusion(
                    n_channels_encoder=n_skip,
                    n_channels_decoder=n_skip,    # avoids conv in fusion
                    activation=None,
                    normalization=None
                )
            )
        self.skip_fusions = nn.ModuleList(skip_fusions)

        skip_branches = []
        for i, (n_skip, n_dec) in enumerate(zip(fusion_n_channels,
                                                n_channels[1:])):
            # determine scale factor for upsampling
            cur_downsampling = fusion_downsamplings[i]
            scale_factor = cur_downsampling // downsampling_in_heads

            skip_branches.append(nn.Sequential(
                OrderedDict([
                    ('embedding', ConvNormAct(
                        n_channels_in=n_skip,
                        n_channels_out=n_dec,
                        kernel_size=1,
                        normalization=None,
                        activation=None
                    )),
                    ('upsample', upsampling(
                        n_channels=n_dec,
                        scale_factor=scale_factor
                    ))
                ])
            ))
        self.skip_branches = nn.ModuleList(skip_branches)

        # final convolution after concatenation
        self.fuse = ConvNormAct(
            n_channels_in=sum(n_channels),
            n_channels_out=sum(n_channels)//len(n_channels),
            kernel_size=1,
            normalization=normalization,    # should be batchnorm
            activation=activation    # should be relu
        )
        self.dropout = nn.Dropout2d(dropout_p)

    def _forward_training(
        self,
        x: DecoderInputType,
        skips: EncoderSkipsType
    ) -> DecoderRawOutputType:
        # note all inputs must be NCHW

        # we are only interested in the main output, not in context features
        x, _ = x

        # main branch
        features = [self.main_branch(x)]

        # skip connection branches from encoder
        for i, (fusion, branch) in enumerate(zip(self.skip_fusions,
                                                 self.skip_branches)):
            # get skip connection with correct downsampling
            # string casting is used to prevent casting keys from int to
            # tensor(int) while exporting to ONNX
            skip = skips[str(self._fusion_downsamplings[i])]
            # apply fusion (actually it is a selection, we do not have decoder
            # features here)
            x_fused = fusion(x_enc=skip, x_dec=None)
            features.append(branch(x_fused))

        # apply final conv on concatenated features
        x = self.fuse(torch.cat(features, dim=1))

        # apply dropout
        x = self.dropout(x)

        # apply task head and final upsampling
        output = self.task_head(x)

        # we do not have side outputs in MLP decoders
        return output, ()

    @property
    @abc.abstractmethod
    def task_head(self) -> nn.Module:
        pass
