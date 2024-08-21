# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch import Tensor
from torch.nn.functional import interpolate

from ..utils import partial_class


KNOWN_UPSAMPLING_METHODS = (
    'nearest',    # nearest interpolation
    'bilinear',    # bilinear interpolation
    'learned-3x3',    # nearest + reflection padding + depth-wise conv
    'learned-3x3-zeropad'    # nearest + zero padding + depth-wise conv
)


class Upsampling(nn.Module):
    def __init__(
        self,
        mode: str,
        n_channels: int,
        scale_factor: Union[float, Tuple[float, float]] = 2.,
    ) -> None:
        super().__init__()

        if mode == 'bilinear':
            self._align_corners = False
        else:
            self._align_corners = None

        if 'learned-3x3' in mode:
            # mimics bilinear interpolation by first applying nearest neighbor
            # upscaling and subsequently a 3x3 depthwise conv with weights
            # realizing bilinear interpolation
            # note, only works as supposed when feature maps are upsampled by
            # a factor of 2
            assert scale_factor == 2. or scale_factor == (2., 2.)

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(n_channels, n_channels,
                                      groups=n_channels,
                                      kernel_size=3,
                                      padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(n_channels, n_channels,
                                      groups=n_channels,
                                      kernel_size=3,
                                      padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w]*n_channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self._mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self._mode = mode

        self._scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        # note that recently, onnx op requires a scale parameter
        # _, _, h, w = x.shape
        x = interpolate(x,
                        # size=(int(h*2), int(w*2)),
                        scale_factor=self._scale_factor,
                        mode=self._mode,
                        align_corners=self._align_corners)
        x = self.pad(x)
        x = self.conv(x)

        return x


UpsamplingType = Upsampling


def get_upsampling_class(
    name: Optional[str] = None,
    **kwargs: Any
) -> Type[UpsamplingType]:
    # global default
    if name is None:
        name = 'bilinear'

    name = name.lower()
    if name not in KNOWN_UPSAMPLING_METHODS:
        raise ValueError(f"Unknown upsampling: '{name}'")
    kwargs['mode'] = name

    return partial_class(Upsampling, **kwargs)
