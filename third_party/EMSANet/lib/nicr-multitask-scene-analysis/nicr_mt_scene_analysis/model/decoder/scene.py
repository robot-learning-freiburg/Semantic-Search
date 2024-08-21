# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Type

import torch
from torch import nn
import torch.nn.functional as F

from ...types import DecoderInputType
from ...types import EncoderSkipsType
from ...types import DecoderRawOutputType
from ..postprocessing import get_postprocessing_class
from ..postprocessing import PostProcessingType
from .base import DecoderBase


class SceneClassificationDecoder(DecoderBase):
    def __init__(
        self,
        n_channels_in: int,
        n_classes: int,
        postprocessing: Type[PostProcessingType] = get_postprocessing_class('scene'),
        **kwargs: Any    # just to catch unused parameters
    ) -> None:
        super().__init__(postprocessing=postprocessing)

        self._task_head = nn.Linear(n_channels_in, n_classes)

    def _forward_training(
        self,
        x: DecoderInputType,
        skips: EncoderSkipsType,
    ) -> DecoderRawOutputType:
        # we are only interested in the first context feature of the context
        # module (the global average pooling branch)
        # if there is no context module, the output of the (dummy) context
        # module (with adapted number of channels) is global average pooled
        cm_output, cm_context_features = x

        if cm_context_features:
            # use the global average pooling branch
            x = cm_context_features[0]

            if x.shape[-2:] != (1, 1):
                # when training with appm context module and processing larger
                # inputs during validation x might be larger than (1, 1)
                # we apply another avg. pooling to make sure that x is (1, 1)
                # note that this chained avg pooling is mathematically
                # equivalent to performing a single global avg. pooling
                x = F.adaptive_avg_pool2d(x, 1)
        else:
            # apply global average pooling to the output
            # use fallback if there is no context module, note that this may
            # change the number of channels compared to training with context
            # module
            x = F.adaptive_avg_pool2d(cm_output, 1)

        # apply fully-connected layer
        x = torch.flatten(x, 1)
        out = self._task_head(x)

        return out, None   # there are no side outputs
