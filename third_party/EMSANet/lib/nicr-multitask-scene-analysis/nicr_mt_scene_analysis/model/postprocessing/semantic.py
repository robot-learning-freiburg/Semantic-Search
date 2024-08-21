# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
import torch.nn.functional as F

from ...data.preprocessing.resize import get_fullres_key, get_fullres_shape
from ...types import BatchType, DecoderRawOutputType, PostprocessingOutputType
from .dense_base import DensePostprocessingBase


class SemanticPostprocessing(DensePostprocessingBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.evidential_output = kwargs.get("evidential_output", False)

    def _postprocess_training(
        self, data: DecoderRawOutputType, batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outputs
        output, side_outputs = data

        # create results dict
        r_dict = {"semantic_output": output, "semantic_side_outputs": side_outputs}

        return r_dict

    def _postprocess_inference(
        self, data: DecoderRawOutputType, batch: BatchType
    ) -> PostprocessingOutputType:
        # unpack outpus (there are no side outputs)
        output, side_outputs = data

        # create results dict
        r_dict = {"semantic_output": output, "semantic_side_outputs": side_outputs}

        # apply softmax and get max / argmax
        if self.evidential_output:
            evidence = F.softplus(output) + 1
            pred = evidence / evidence.sum(dim=1, keepdim=True)
        else:
            pred = F.softmax(output, dim=1)
        score, idx = torch.max(pred, dim=1)

        r_dict.update(
            {
                "semantic_softmax_scores": pred,
                "semantic_segmentation_score": score,
                "semantic_segmentation_idx": idx,
            }
        )

        # resize output to original shape (assume same shape for all samples)
        shape = get_fullres_shape(batch, "semantic")

        if shape != tuple(output.shape[-2:]):
            # we have to resize prediction
            output_fullres = self._resize_prediction(
                output, shape=shape, mode="bilinear"
            )
            pred_fullres = F.softmax(output_fullres, dim=1)
            score_fullres, idx_fullres = torch.max(pred_fullres, dim=1)
        else:
            # no resize necessary, save that ressources
            output_fullres = output.clone()
            pred_fullres = pred.clone()
            score_fullres = score.clone()
            idx_fullres = idx.clone()

        # update results dict
        r_dict.update(
            {
                get_fullres_key("semantic_output"): output_fullres,
                get_fullres_key("semantic_softmax_scores"): pred_fullres,
                get_fullres_key("semantic_segmentation_score"): score_fullres,
                get_fullres_key("semantic_segmentation_idx"): idx_fullres,
            }
        )

        return r_dict
