# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>

TODO:
- add tests, currently only partially tested in decoder tests and
  EMSANet/EMSAFormer repos
"""
from typing import Any, Callable, Optional, Type, Union

import torch
from torch import Tensor
import torch.nn as nn

from ..utils import partial_class
from .activation import get_activation_class
from ..types import EncoderSkipType
from .normalization import get_normalization_class
from .utils import ConvNormAct


KNOWN_ENCODER_DECODER_FUSIONS = (
    # add encoder or rgb/depth encoder features to decoder features
    'add', 'add-rgb', 'add-depth',
    # select encoder or rgb/depth encoder features, ignore decoder features
    # (useful for MLP decoder)
    'select', 'select-rgb', 'select-depth',
    # according to swin implementation: with add. layernorm and NHWC -> NCHW
    'swin-ln-add', 'swin-ln-add-rgb', 'swin-ln-add-depth',
    'swin-ln-select', 'swin-ln-select-rgb', 'swin-ln-select-depth',
    # swin without layernorm
    'swin-add', 'swin-add-rgb', 'swin-add-depth',
    'swin-select', 'swin-select-rgb', 'swin-select-depth',
    # do not fuse features from encoder, simply return decoder features
    'none'
)


class EncoderDecoderFusion(nn.Module):
    def __init__(
        self,
        n_channels_encoder: int,
        n_channels_decoder: int,
        fuse_features_from: Union[str, None],
        fuse_operation: Union[Callable[[Tensor, Tensor], Tensor], None] = torch.add,
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class()
    ) -> None:
        """Generic encoder-decoder fusion module."""
        super().__init__()

        if fuse_operation is not None:
            if n_channels_encoder != n_channels_decoder:
                # adapt number of channels if both numbers differ
                self.layer = ConvNormAct(n_channels_encoder,
                                         n_channels_decoder,
                                         normalization=normalization,
                                         activation=activation)
            else:
                # nothing to do, do not add any parameters here
                self.layer = nn.Identity()

        self._fuse_features_from = fuse_features_from
        self._fuse_operation = fuse_operation

    def forward(
        self,
        x_enc: EncoderSkipType,
        x_dec: Union[Tensor, None]
    ) -> Tensor:
        if self._fuse_operation is None:
            # encoder decoder fusion is disabled, ignore encoder features
            return x_dec

        if self._fuse_features_from is None:
            # we did not know the key to access the encoder features at
            # creation time, however, there is only one key in the dict,
            # assign it lazily
            assert len(x_enc) == 1
            self._fuse_features_from = list(x_enc.keys())[0]

        # apply fusion
        out = self._fuse_operation(
            self.layer(x_enc[self._fuse_features_from]), x_dec
        )

        return out


class EncoderDecoderFusionSwin(EncoderDecoderFusion):
    def __init__(
        self,
        n_channels_encoder: int,
        n_channels_decoder: int,
        fuse_features_from: Union[str, None],
        fuse_operation: Callable[[Tensor, Tensor], Tensor] = torch.add,
        apply_layer_norm: bool = True,
        normalization: Type[nn.Module] = get_normalization_class(),
        activation: Type[nn.Module] = get_activation_class()
    ) -> None:
        """Encoder-decoder fusion module for SwinTransformers with LN"""
        super().__init__(
            n_channels_encoder=n_channels_encoder,
            n_channels_decoder=n_channels_decoder,
            fuse_features_from=fuse_features_from,
            fuse_operation=fuse_operation,
            normalization=normalization,
            activation=activation
        )

        # Swin transformers apply an additional layer norm to the features
        # of each stage before using encoder-decoder fusion, moreover,
        # channels_last (nhwc) is used for the encoder features, this class
        # handles both
        if apply_layer_norm:
            # note that we fuse only a single modality, so a single
            # layernorm here is enough
            self.ln = get_normalization_class('ln')(n_channels_encoder)
        else:
            self.ln = nn.Identity()

    def forward(
        self,
        x_enc: EncoderSkipType,
        x_dec: Union[Tensor, None]
    ) -> Tensor:
        # note, encoder inputs are NHWC (channels last) for SwinTransformers,
        # while the decoders use NCHW

        if self._fuse_features_from is None:
            # we did not know the key to access the encoder features at
            # creation time, however, there is only one key in the dict,
            # assign it lazily
            assert len(x_enc) == 1
            self._fuse_features_from = list(x_enc.keys())[0]

        # preprocess encoder features
        x_ = x_enc[self._fuse_features_from]

        # opt: apply layer norm
        x_ = self.ln(x_)

        # force NCHW for decoder
        x_ = torch.permute(x_, (0, 3, 1, 2))

        # apply fusion
        return super().forward({self._fuse_features_from: x_}, x_dec)


EncoderDecoderFusionType = Union[EncoderDecoderFusion, EncoderDecoderFusionSwin]


def get_encoder_decoder_fusion_class(
    name: Optional[str] = None,
    **kwargs: Any
) -> Type[EncoderDecoderFusionType]:
    # global default
    if name is None:
        name = 'add-rgb'

    name = name.lower()
    if name not in KNOWN_ENCODER_DECODER_FUSIONS:
        raise ValueError(f"Unknown encoder decoder fusion: '{name}'")

    if 'none' == name:
        # no encoder-decoder fusion
        kwargs['fuse_features_from'] = None
        kwargs['fuse_operation'] = None
        return partial_class(EncoderDecoderFusion, **kwargs)

    # determine fusion type
    if 'swin-ln' in name:
        cls = EncoderDecoderFusionSwin
        kwargs['apply_layer_norm'] = True
    elif 'swin' in name:
        cls = EncoderDecoderFusionSwin
        kwargs['apply_layer_norm'] = False
    else:
        cls = EncoderDecoderFusion

    # determine fusion operation
    if 'add' in name:
        # add encoder add decoder features
        kwargs['fuse_operation'] = torch.add
    elif 'select' in name:
        # select encoder features, ignore encoder features (useful for MLP
        # decoder)
        kwargs['fuse_operation'] = lambda x_enc, x_dec: x_enc
    else:
        raise ValueError(f"Unknown encoder decoder fusion operation.")

    # determine which features to fuse
    if 'rgb' in name:
        kwargs['fuse_features_from'] = 'rgb'
    elif 'depth' in name:
        kwargs['fuse_features_from'] = 'depth'
    else:
        # no specific modality to fuse -> single backbone/modality, determine
        # key later
        kwargs['fuse_features_from'] = None

    return partial_class(cls, **kwargs)
