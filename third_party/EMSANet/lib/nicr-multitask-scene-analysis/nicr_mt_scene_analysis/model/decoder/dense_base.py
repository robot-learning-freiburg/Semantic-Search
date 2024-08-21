# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import abc
from typing import Optional, Tuple, Type

import torch.nn as nn
from torch import Tensor

from ...types import DecoderInputType
from ...types import DenseDecoderModuleOutputType
from ...types import DecoderRawOutputType
from ...types import EncoderSkipsType
from ..activation import get_activation_class
from ..block import BlockType
from ..encoder_decoder_fusion import EncoderDecoderFusionType
from ..normalization import get_normalization_class
from ..postprocessing import PostProcessingType
from ..upsampling import get_upsampling_class
from ..upsampling import UpsamplingType
from ..utils import ConvNormAct
from .base import DecoderBase


class DenseDecoderModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels: int,
        block: Type[BlockType],
        n_blocks: int,
        initial_conv: bool = True,
        activation: Type[nn.Module] = get_activation_class(),
        normalization: Type[nn.Module] = get_normalization_class(),
        upsampling: Optional[Type[UpsamplingType]] = get_upsampling_class()
    ) -> None:
        super().__init__()

        # initial conv
        if initial_conv:
            self.conv = ConvNormAct(n_channels_in, n_channels,
                                    kernel_size=3,
                                    normalization=normalization,
                                    activation=activation)
            blocks_n_channels = [n_channels] * (n_blocks+1)
        else:
            self.conv = nn.Identity()
            blocks_n_channels = [n_channels_in] + [n_channels] * n_blocks
            assert n_blocks > 0

        # subsequent blocks
        blocks = []
        for i in range(n_blocks):
            if blocks_n_channels[i] != blocks_n_channels[i+1]:
                # skip connections need to modify the number of channels
                downsample = ConvNormAct(
                    blocks_n_channels[i], blocks_n_channels[i+1],
                    kernel_size=1,
                    activation=None
                )
            else:
                downsample = None

            blocks.append(
                block(inplanes=blocks_n_channels[i],
                      planes=blocks_n_channels[i+1],
                      stride=1,
                      downsample=downsample,
                      groups=1,
                      base_width=64,
                      dilation=1,
                      normalization=normalization,
                      activation=activation)
            )
        self.blocks = nn.Sequential(*blocks)

        # final upsampling
        if upsampling is not None:
            # perform upsampling (by factor of 2)
            self.upsample = upsampling(n_channels=n_channels)
        else:
            # skip upsampling
            self.upsample = nn.Identity()

    def forward(self, x: Tensor) -> DenseDecoderModuleOutputType:
        out = self.conv(x)
        out = self.blocks(out)

        # for multiscale supervision
        out_side = out if self.training else None

        out = self.upsample(out)

        return out, out_side


class DenseDecoderBase(DecoderBase):
    def __init__(
        self,
        n_channels_in: int,
        downsampling_in: int,
        n_channels: Tuple[int, ...],
        downsamplings: Tuple[int, ...],
        block: Type[BlockType],
        n_blocks: int,
        fusion: Type[EncoderDecoderFusionType],
        fusion_n_channels: Tuple[int, ...],
        fusion_downsamplings: Tuple[int, ...],
        postprocessing: Type[PostProcessingType],
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: Type[UpsamplingType] = get_upsampling_class()
    ) -> None:
        super().__init__(postprocessing=postprocessing)

        # perform some simple sanity checks
        assert len(n_channels) == len(downsamplings)
        assert sorted(downsamplings, reverse=True) == list(downsamplings)
        assert all(d <= downsampling_in for d in downsamplings)

        assert len(fusion_n_channels) == len(fusion_downsamplings)
        assert sorted(fusion_downsamplings,
                      reverse=True) == list(fusion_downsamplings)

        # create decoder and fusion modules
        # note, the number of decoder modules is determined by
        # len(n_channels) / len(downsamplings), moreover, note that we create
        # side outputs only if downsampling is decreased
        cur_downsampling = downsampling_in
        decoder_modules = []
        fusions = []

        side_output_downscales = []
        side_output_n_channels = []
        decoder_modules_consider_side_output = []
        decoder_modules_fusion_downsamplings = []

        n_dec_in = (n_channels_in,) + n_channels[:-1]
        n_dec_out = n_channels

        for i in range(len(n_channels)):
            # decoder module
            n_in = n_dec_in[i]
            n_out = n_dec_out[i]
            ds = downsamplings[i]

            if ds < cur_downsampling:
                # decoder module should perform upsampling, create side output
                # before upsampling
                decoder_modules_consider_side_output.append(True)
                side_output_downscales.append(cur_downsampling)
                side_output_n_channels.append(n_out)

                do_upsampling = True
                cur_downsampling = ds
            else:
                # no upsampling, no side output required
                decoder_modules_consider_side_output.append(False)

                do_upsampling = False

            decoder_modules.append(
                DenseDecoderModule(
                    n_channels_in=n_in,
                    n_channels=n_out,
                    block=block,
                    n_blocks=n_blocks,
                    activation=activation,
                    normalization=normalization,
                    upsampling=upsampling if do_upsampling else None
                )
            )

            # fusion
            if cur_downsampling in fusion_downsamplings:
                # encoder features should be fused
                decoder_modules_fusion_downsamplings.append(cur_downsampling)

                n_skip = fusion_n_channels[len(fusions)]
                fusions.append(
                    fusion(
                        n_channels_encoder=n_skip,
                        n_channels_decoder=n_out,
                        activation=activation,
                        normalization=normalization
                    )
                )
            else:
                # encoder features should NOT be fused
                decoder_modules_fusion_downsamplings.append(-1)

        # convert to nn.ModuleList
        self.decoder_modules = nn.ModuleList(decoder_modules)
        self._side_output_downscales = tuple(side_output_downscales)
        self._side_output_n_channels = tuple(side_output_n_channels)
        self._decoder_modules_consider_side_output = tuple(
            decoder_modules_consider_side_output
        )

        self.fusions = nn.ModuleList(fusions)
        self._decoder_modules_fusion_downsamplings = tuple(
            decoder_modules_fusion_downsamplings
        )

    @property
    @abc.abstractmethod
    def task_head(self) -> nn.Module:
        pass

    @property
    @abc.abstractmethod
    def side_output_heads(self) -> nn.ModuleList:
        pass

    @property
    def side_output_downscales(self) -> Tuple[int, ...]:
        return self._side_output_downscales

    @property
    def side_output_n_channels(self) -> Tuple[int, ...]:
        return self._side_output_n_channels

    def _forward_decoder_modules(
        self,
        x: Tensor,
        skips: EncoderSkipsType,
    ) -> DecoderRawOutputType:
        # some checks
        assert len(skips) == len(self.fusions)

        side_outputs = []
        fusion_idx = 0
        for i in range(len(self.decoder_modules)):

            # apply decoder module
            dec_m = self.decoder_modules[i]
            consider_side_output = self._decoder_modules_consider_side_output[i]

            x, side_outs = dec_m(x)
            if consider_side_output:
                side_outputs.append(side_outs)

            # apply encoder-decoder fusion
            fusion_ds = self._decoder_modules_fusion_downsamplings[i]
            if -1 != fusion_ds:
                fusion = self.fusions[fusion_idx]
                # string casting is used to prevent casting keys from int to
                # tensor(int) while exporting to ONNX
                x = fusion(x_enc=skips[str(fusion_ds)], x_dec=x)

                fusion_idx += 1

        return x, tuple(side_outputs)

    def _forward_training(
        self,
        x: DecoderInputType,
        skips: EncoderSkipsType,
    ) -> DecoderRawOutputType:
        # we are only interested in the main output, not in the context features
        x, _ = x

        # apply decoder modules
        output, side_outputs = self._forward_decoder_modules(
            x=x,
            skips=skips
        )

        # apply task head and final upsampling
        output = self.task_head(output)

        # apply side output heads
        side_outputs = [
            side_head(side_out) if side_out is not None else None
            for side_head, side_out in zip(self.side_output_heads, side_outputs)
        ]

        return output, tuple(side_outputs)
