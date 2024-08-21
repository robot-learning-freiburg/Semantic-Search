# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional, Sequence, Tuple, Type, Union

import abc
from collections import defaultdict
from itertools import compress

import torch
from torch import nn

from ..types import EncoderInputType
from ..types import EncoderOutputType
from .activation import get_activation_class
from .backbone.base import Backbone
from .encoder_fusion import EncoderFusionType
from .encoder_fusion import get_encoder_fusion_class
from .normalization import get_normalization_class


class EncoderBase(abc.ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abc.abstractmethod
    def skips_n_channels(self) -> Tuple[int]:
        """The number of channels in each skip connection"""
        pass

    @property
    @abc.abstractmethod
    def skips_downsamplings(self) -> Sequence[int]:
        """The downsampling factors of each skip connection"""
        pass

    @property
    @abc.abstractmethod
    def n_channels_out(self) -> int:
        """The number of output channels of the encoder"""
        pass

    @property
    @abc.abstractmethod
    def downsampling(self):
        """The total downsampling factor at the end of the encoder"""
        pass

    @abc.abstractmethod
    def forward(self, x: EncoderInputType) -> EncoderOutputType:
        # The encoder gets a dictionary of tensors, with the key indicating
        # the modality and the value the torch tensor for processing. The
        # encoder processes this input using a single or multiple backbones
        # and returns a tuple of two dictionaries. The first dictionary
        # contains the output of the backbone and the second dictionary
        # contains the skip connections. Both dictionaries should have the same
        # keys as the input.
        pass


class Encoder(EncoderBase):
    def __init__(
        self,
        backbone: Union[Backbone, None],
        skip_downsamplings: Sequence[int] = (4, 8, 16)
    ) -> None:
        """Single-backbone encoder"""

        super().__init__()

        # store backbone
        self.backbone = backbone

        # compute some stuff for forward propagation
        self._n_stages = len(self.backbone.stages)
        self._stages_downsampling = self.backbone.stages_downsampling
        self._stages_n_channels = self.backbone.stages_n_channels

        # determine where to create skip connections
        self._skips_downsamplings = skip_downsamplings
        self._stages_skip_connections = [False] * self._n_stages
        for ds in self._skips_downsamplings:
            last_idx = self._stages_downsampling[::-1].index(ds)
            stage_idx = self._n_stages - 1 - last_idx

            # TODO(dase+sofi): make this nicer
            if stage_idx == (self._n_stages - 1):
                # we hit the last stage, e.g., due to reduced downsampling, use
                # first occurrence instead
                stage_idx = self._stages_downsampling.index(ds)

            self._stages_skip_connections[stage_idx] = True

    @property
    def skips_n_channels(self) -> Tuple[int]:
        return tuple(compress(data=self._stages_n_channels,
                              selectors=self._stages_skip_connections))

    @property
    def skips_downsamplings(self) -> Sequence[int]:
        return self._skips_downsamplings

    @property
    def n_channels_out(self) -> int:
        return self._stages_n_channels[-1]

    @property
    def downsampling(self):
        return self._stages_downsampling[-1]

    def forward(self, x: EncoderInputType) -> EncoderOutputType:
        # the encoder gets a single input in a dictionary
        assert len(x) == 1

        key = list(x.keys())[0]
        x_ = x[key]

        # feature for skip connections
        decoder_skips_dict = defaultdict(dict)

        # forward stages
        downsampling_idx = 0
        for idx in range(self._n_stages):
            # forward stage
            x_ = self.backbone.forward_stage(idx, x_)

            # check if features for a skip connection need to be stored
            if self._stages_skip_connections[idx]:
                # skip connections are also stored in a dictionary with:
                # - key: the downsampling factor as string (string is used to
                #   prevent converting keys from int to tensor(int) during
                #   ONNX export)
                # - value: the features as dictionary similar to
                #   EncoderInputType, i.e., with key: modality and value:
                #   features for this modality
                cur_downsampling = self.skips_downsamplings[downsampling_idx]
                decoder_skips_dict[str(cur_downsampling)][key] = x_
                downsampling_idx += 1

        return {key: x_}, dict(decoder_skips_dict)


class FusedRGBDEncoder(EncoderBase):
    def __init__(
        self,
        backbone_rgb: Union[Backbone, None],
        backbone_depth: Union[Backbone, None],
        fusion: Type[EncoderFusionType],
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class(),
        skip_downsamplings: Sequence[int] = (4, 8, 16)
    ) -> None:
        """Dual-backbone encoder for processing RGB-D"""
        super().__init__()

        # store backbones
        self.backbone_rgb = backbone_rgb
        self.backbone_depth = backbone_depth

        # create fusion modules for RGBD
        if self.backbone_rgb is not None and self.backbone_depth is not None:
            # check some assumptions
            b_rgb = self.backbone_rgb
            b_depth = self.backbone_depth
            assert len(b_rgb.stages) == len(b_depth.stages)
            assert b_rgb.stages_n_channels == b_depth.stages_n_channels
            assert b_rgb.stages_downsampling == b_depth.stages_downsampling
            assert b_rgb.stages_memory_layout == b_depth.stages_memory_layout

            fusions = []
            for n, memory_layout in zip(b_rgb.stages_n_channels,
                                        b_rgb.stages_memory_layout):
                fusions.append(fusion(n_channels_in=n,
                                      normalization=normalization,
                                      activation=activation,
                                      input_memory_layout=memory_layout))
            self.fusions = nn.ModuleList(fusions)

        # compute some stuff for forward propagation
        bb = self.backbone_rgb or self.backbone_depth
        self._n_stages = len(bb.stages)
        self._stages_downsampling = bb.stages_downsampling
        self._stages_n_channels = bb.stages_n_channels

        # determine where to create skip connections
        self._skips_downsamplings = skip_downsamplings
        self._stages_skip_connections = [False] * self._n_stages
        for ds in self._skips_downsamplings:
            last_idx = self._stages_downsampling[::-1].index(ds)
            stage_idx = self._n_stages - 1 - last_idx

            # TODO(dase+sofi): make this nicer
            if stage_idx == (self._n_stages - 1):
                # we hit the last stage, e.g., due to reduced downsampling, use
                # first occurrence instead
                stage_idx = self._stages_downsampling.index(ds)

            self._stages_skip_connections[stage_idx] = True

    @property
    def skips_n_channels(self) -> Tuple[int]:
        return tuple(compress(data=self._stages_n_channels,
                              selectors=self._stages_skip_connections))

    @property
    def skips_downsamplings(self) -> Sequence[int]:
        return self._skips_downsamplings

    @property
    def n_channels_out(self) -> int:
        return self._stages_n_channels[-1]

    @property
    def downsampling(self):
        return self._stages_downsampling[-1]

    def forward(self, x: EncoderInputType) -> EncoderOutputType:
        # the RGB-D encoder comprises two backbones and takes two tensors, i.e.,
        # rgb and depth as input
        assert len(x) == 2

        # feature for skip connections
        decoder_skips_dict = defaultdict(dict)

        # forward stages
        backbone_rgb = self.backbone_rgb
        backbone_depth = self.backbone_depth
        downsampling_idx = 0
        x_ = {'rgb': x['rgb'], 'depth': x['depth']}  # do not modify input dict
        for idx in range(self._n_stages):
            # forward rgb stage
            if backbone_rgb is not None:
                x_['rgb'] = backbone_rgb.forward_stage(idx, x_['rgb'])

            # forward depth stage
            if backbone_depth is not None:
                x_['depth'] = backbone_depth.forward_stage(idx, x_['depth'])

            # fuse both outputs
            x_ = self.fusions[idx](x_)

            # check if features for a skip connection need to be stored
            if self._stages_skip_connections[idx]:
                # skip connections are also stored in a dictionary with:
                # - key: the downsampling factor as string (string is used to
                #   prevent converting keys from int to tensor(int) during
                #   ONNX export)
                # - value: the features as dictionary similar to
                #   EncoderInputType, i.e., with key: modality and value:
                #   features for this modality
                cur_downsampling = self.skips_downsamplings[downsampling_idx]
                decoder_skips_dict[str(cur_downsampling)]['rgb'] = x_['rgb']
                decoder_skips_dict[str(cur_downsampling)]['depth'] = x_['depth']
                downsampling_idx += 1

        return x_, dict(decoder_skips_dict)


EncoderType = Union[Encoder, FusedRGBDEncoder]


def get_encoder(
    backbone_rgb: Union[Backbone, None] = None,
    backbone_depth: Union[Backbone, None] = None,
    backbone_rgbd: Union[Backbone, None] = None,
    fusion: Optional[str] = None,
    normalization: str = 'batchnorm',
    activation: str = 'relu',
    skip_downsamplings: Sequence[int] = (4, 8, 16)
) -> EncoderType:
    # create encoder(s), i.e., fused rgb+depth, rgbd (rgb+depth with a single
    # backbone), or single modality: rgb or depth

    if (backbone_rgb is not None and backbone_depth is not None):
        # fused dual-modality RGB-D encoder
        return FusedRGBDEncoder(
            backbone_rgb=backbone_rgb,
            backbone_depth=backbone_depth,
            fusion=get_encoder_fusion_class(fusion),
            normalization=get_normalization_class(normalization),
            activation=get_activation_class(activation),
            skip_downsamplings=skip_downsamplings
        )

    elif backbone_rgbd is not None:
        # rgbd (rgb+depth with a single backbone)
        backbone = backbone_rgbd

    elif (backbone_rgb is not None) ^ (backbone_depth is not None):
        # single modality: rgb or depth
        backbone = backbone_rgb or backbone_depth

    else:
        raise ValueError('Either `backbone_rgb` and/or `backbone_depth` or '
                         '`backbone_rgbd` must be given.')

    return Encoder(backbone=backbone, skip_downsamplings=skip_downsamplings)
