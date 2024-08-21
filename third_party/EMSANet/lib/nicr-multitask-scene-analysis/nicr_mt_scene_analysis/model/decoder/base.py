# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import abc
from typing import Tuple, Type

import torch.nn as nn

from ..postprocessing import PostProcessingType
from ...types import BatchType
from ...types import DecoderInputType
from ...types import DecoderOutputType
from ...types import DecoderRawOutputType
from ...types import EncoderSkipsType


class DecoderBase(abc.ABC, nn.Module):
    def __init__(
        self,
        postprocessing: Type[PostProcessingType]
    ) -> None:
        super().__init__()

        self._postprocessing = postprocessing()

    @property
    def side_output_downscales(self) -> Tuple:
        return ()

    @property
    def postprocessing(self):
        return self._postprocessing

    def forward(
        self,
        x: DecoderInputType,
        skips: EncoderSkipsType,
        batch: BatchType,
        do_postprocessing: bool = True
    ) -> DecoderOutputType:
        # apply decoder
        if self.training:
            output = self._forward_training(x, skips)
        else:
            output = self._forward_inference(x, skips)

        # apply postprocessing
        if do_postprocessing:
            output = self._postprocessing.postprocess(
                output, batch,
                is_training=self.training
            )

        return output

    @abc.abstractmethod
    def _forward_training(
        self,
        x: DecoderInputType,
        skips: EncoderSkipsType,
    ) -> DecoderRawOutputType:
        pass

    def _forward_inference(
        self,
        x: DecoderInputType,
        skips: EncoderSkipsType,
    ) -> DecoderRawOutputType:
        # default behavior: same as in training
        return self._forward_training(x, skips)
