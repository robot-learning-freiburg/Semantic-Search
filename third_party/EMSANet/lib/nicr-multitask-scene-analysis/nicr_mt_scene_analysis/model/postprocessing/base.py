# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import abc

from ...types import BatchType
from ...types import DecoderRawOutputType
from ...types import PostprocessingOutputType


class PostprocessingBase(abc.ABC):
    def postprocess(
        self,
        data: DecoderRawOutputType,
        batch: BatchType,
        is_training: bool = True
    ) -> PostprocessingOutputType:

        if is_training:
            return self._postprocess_training(data, batch)
        else:
            return self._postprocess_inference(data, batch)

    @abc.abstractmethod
    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        pass

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # default behavior: same as in training
        return self._postprocess_training(data, batch)
