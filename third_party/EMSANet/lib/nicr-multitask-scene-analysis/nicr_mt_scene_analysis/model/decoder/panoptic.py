# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Tuple

from torch import nn

from ..postprocessing import get_postprocessing_class
from ..postprocessing import PostProcessingType
from ...types import BatchType
from ...types import DecoderInputType
from ...types import DecoderPostprocessedOutputType
from ...types import EncoderSkipsType
from .instance import InstanceDecoder
from .semantic import SemanticDecoder


class PanopticHelper(nn.Module):
    def __init__(
        self,
        semantic_decoder: SemanticDecoder,
        instance_decoder: InstanceDecoder,
        postprocessing: PostProcessingType = get_postprocessing_class('panoptic'),
    ) -> None:
        super().__init__()
        self.semantic_decoder = semantic_decoder
        self.instance_decoder = instance_decoder

        # determine combined postprocessing
        self._postprocessing = postprocessing()

    @property
    def side_output_downscales(self) -> Tuple[int]:
        scales = set(self.semantic_decoder.side_output_downscales)
        scales |= set(self.instance_decoder.side_output_downscales)
        return tuple(scales)

    @property
    def postprocessing(self):
        return self._postprocessing

    def forward(
        self,
        x: DecoderInputType,
        skips: EncoderSkipsType,
        batch: BatchType,
        do_postprocessing: bool = True
    ) -> DecoderPostprocessedOutputType:
        # forward semantic decoder
        res_semantic = self.semantic_decoder.forward(
            x=x,
            skips=skips,
            batch=batch,
            do_postprocessing=False     # skip postprocessing, it is done later
        )
        # forward instance decoder
        res_instance = self.instance_decoder.forward(
            x=x,
            skips=skips,
            batch=batch,
            do_postprocessing=False     # skip postprocessing, it is done later
        )

        # combine both outputs
        s_output, s_side_outputs = res_semantic
        i_output, i_side_outputs = res_instance
        output = (s_output, i_output), (s_side_outputs, i_side_outputs)

        # apply postprocessing
        if do_postprocessing:
            output = self._postprocessing.postprocess(output, batch,
                                                      is_training=self.training)

        return output
