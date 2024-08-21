# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from math import log2
from typing import Optional, Tuple, Type

import torch
from torch import nn
from torch import Tensor

from ...utils import OrientationOutputNormalization
from ..activation import get_activation_class
from ..block import BlockType
from ..normalization import get_normalization_class
from ..encoder_decoder_fusion import EncoderDecoderFusionType
from ..upsampling import get_upsampling_class
from ..upsampling import UpsamplingType
from ..utils import ConvNormAct
from ..postprocessing import get_postprocessing_class
from ..postprocessing import PostProcessingType
from .dense_base import DenseDecoderBase
from .mlp_base import MLPDecoderBase


class InstanceHead(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_per_task: int = 32,    # default panoptic deeplab
        with_orientation: bool = False,
        sigmoid_for_center: bool = True,
        tanh_for_offset: bool = True,
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: Optional[Type[UpsamplingType]] = None,
        n_upsamplings: int = 0
    ) -> None:
        super().__init__()

        # determine number of tasks (center+offset(+orientation)
        n_tasks = 3 if with_orientation else 2

        self._n_tasks = n_tasks
        self._n_channels_per_task = n_channels_per_task
        self._sigmoid_for_center = sigmoid_for_center
        self._tanh_for_offset = tanh_for_offset

        # shared convolution (output is split subsequently)
        self.shared_conv = ConvNormAct(
            n_channels_in, n_tasks*n_channels_per_task,
            kernel_size=3,    # panoptic deeplab: 5
            normalization=normalization,
            activation=activation
        )

        # convolutions mapping to tasks
        # we use 3x3 conv for the main output and 1x1 convs for side outputs
        is_main_output = n_upsamplings != 0
        kernel_size = 3 if is_main_output else 1

        # convolution for center prediction (one output channel)
        conv_center = nn.Conv2d(n_channels_per_task, 1,
                                kernel_size=kernel_size,
                                padding=(kernel_size-1)//2)

        # convolution for offset prediction (output channel for x and y)
        conv_offset = nn.Conv2d(n_channels_per_task, 2,
                                kernel_size=kernel_size,
                                padding=(kernel_size-1)//2)
        task_convs = [conv_center, conv_offset]

        if self._n_tasks == 3:
            # add convolution for orientation (output channel for sin and cos)
            conv_orientation = nn.Conv2d(n_channels_per_task, 2,
                                         kernel_size=kernel_size,
                                         padding=(kernel_size-1)//2)
            task_convs.append(conv_orientation)
            # normalize output (unit length -- sin^2 + cos^2 = 1)
            self._act_orientation = OrientationOutputNormalization()

        self.task_convs = nn.ModuleList(task_convs)

        # prediction upsampling
        n_channels = 3 if 2 == n_tasks else 5

        upsamplings = []
        for _ in range(n_upsamplings):
            upsamplings.append(
                upsampling(n_channels=n_channels)
            )
        self.upsampling = nn.Sequential(*upsamplings)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        # apply shared convolution
        x = self.shared_conv(x)

        # apply convolution for each task
        outs = []
        for i, conv in enumerate(self.task_convs):
            slice_input = slice(i*self._n_channels_per_task,
                                (i+1)*self._n_channels_per_task)
            outs.append(conv(x[:, slice_input, :, :]))

        # concatenate outs + apply upsampling + split them again
        outs_cat = torch.cat(outs, dim=1)
        outs_cat = self.upsampling(outs_cat)
        outs = torch.split(outs_cat, [int(o.shape[1]) for o in outs], dim=1)

        # apply activation functions
        outs = list(outs)
        if self._sigmoid_for_center:
            outs[0] = torch.sigmoid(outs[0])
        if self._tanh_for_offset:
            outs[1] = torch.tanh(outs[1])

        if self._n_tasks == 3:
            outs[2] = self._act_orientation(outs[2])

        return tuple(outs)


class InstanceDecoder(DenseDecoderBase):
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
        n_channels_per_task: int = 32,
        with_orientation: bool = False,
        sigmoid_for_center: bool = True,
        tanh_for_offset: bool = True,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class('instance'),
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: Type[UpsamplingType] = get_upsampling_class(),
        prediction_upsampling: Type[UpsamplingType] = get_upsampling_class()
    ) -> None:
        super().__init__(n_channels_in=n_channels_in,
                         downsampling_in=downsampling_in,
                         n_channels=n_channels,
                         downsamplings=downsamplings,
                         block=block,
                         n_blocks=n_blocks,
                         fusion=fusion,
                         fusion_n_channels=fusion_n_channels,
                         fusion_downsamplings=fusion_downsamplings,
                         postprocessing=postprocessing,
                         normalization=normalization,
                         activation=activation,
                         upsampling=upsampling)
        # final task head
        self._task_head = InstanceHead(
            n_channels_in=n_channels[-1],
            n_channels_per_task=n_channels_per_task,
            with_orientation=with_orientation,
            sigmoid_for_center=sigmoid_for_center,
            tanh_for_offset=tanh_for_offset,
            normalization=normalization,
            activation=activation,
            upsampling=prediction_upsampling,
            n_upsamplings=int(log2(downsamplings[-1]))
        )

        # heads for side outputs
        side_output_heads = []
        for n in self.side_output_n_channels:
            side_output_heads.append(
                InstanceHead(
                    n_channels_in=n,
                    n_channels_per_task=n_channels_per_task,
                    with_orientation=with_orientation,
                    sigmoid_for_center=sigmoid_for_center,
                    tanh_for_offset=tanh_for_offset,
                    normalization=normalization,
                    activation=activation,
                    upsampling=None,
                    n_upsamplings=0
                )
            )
        self._side_output_heads = nn.ModuleList(side_output_heads)

    @property
    def task_head(self) -> nn.Module:
        return self._task_head

    @property
    def side_output_heads(self) -> nn.ModuleList:
        return self._side_output_heads


class InstanceMLPDecoder(MLPDecoderBase):
    def __init__(
        self,
        n_channels_in: int,
        downsampling_in: int,
        n_channels: Tuple[int, ...],
        fusion: Type[EncoderDecoderFusionType],
        fusion_n_channels: Tuple[int, ...],
        fusion_downsamplings: Tuple[int, ...],
        n_channels_per_task: int = 32,
        with_orientation: bool = False,
        sigmoid_for_center: bool = True,
        tanh_for_offset: bool = True,
        downsampling_in_heads: int = 4,
        dropout_p: float = 0.1,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class('instance'),
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        upsampling: Type[UpsamplingType] = get_upsampling_class(),
        prediction_upsampling: Type[UpsamplingType] = get_upsampling_class()
    ) -> None:
        super().__init__(n_channels_in=n_channels_in,
                         downsampling_in=downsampling_in,
                         n_channels=n_channels,
                         fusion=fusion,
                         fusion_n_channels=fusion_n_channels,
                         fusion_downsamplings=fusion_downsamplings,
                         downsampling_in_heads=downsampling_in_heads,
                         dropout_p=dropout_p,
                         postprocessing=postprocessing,
                         normalization=normalization,
                         activation=activation,
                         upsampling=upsampling)

        self._downsampling_in_heads = downsampling_in_heads

        # final task head
        self._task_head = InstanceHead(
            n_channels_in=sum(n_channels)//len(n_channels),
            n_channels_per_task=n_channels_per_task,
            with_orientation=with_orientation,
            sigmoid_for_center=sigmoid_for_center,
            tanh_for_offset=tanh_for_offset,
            normalization=normalization,
            activation=activation,
            upsampling=prediction_upsampling,
            n_upsamplings=self._downsampling_in_heads//2,   # each doubles res.
        )

    @property
    def task_head(self) -> nn.Module:
        return self._task_head
