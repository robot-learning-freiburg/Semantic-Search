# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
import torch.nn.functional as F

from ...types import BatchType
from ...types import DecoderRawOutputType
from ...types import PostprocessingOutputType
from .base import PostprocessingBase


class ScenePostprocessing(PostprocessingBase):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def _postprocess_training(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs
        output, side_outputs = data

        # create results dict
        r_dict = {
            'scene_output': output
        }

        return r_dict

    def _postprocess_inference(
        self,
        data: DecoderRawOutputType,
        batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs (there are no side outputs)
        output, side_outputs = data

        # apply softmax and get max + argmax
        pred = F.softmax(output, dim=1)
        score, idx = torch.max(pred, dim=1)

        # create results dict
        r_dict = {
            'scene_class_score': score,
            'scene_class_idx': idx,
            'scene_output': output,
        }

        return r_dict
