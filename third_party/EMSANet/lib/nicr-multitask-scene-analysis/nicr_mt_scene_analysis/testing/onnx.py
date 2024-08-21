# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, List, Optional, Tuple

import torch
from torch.onnx import TrainingMode

from . import EXPORT_ONNX_MODELS


def export_onnx_model(
    filepath: str,
    model: torch.nn.Module,
    x: Tuple[Any],
    training_mode: TrainingMode = TrainingMode.PRESERVE,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    force_export: bool = False,
    use_fallback: bool = False
) -> None:
    if not (EXPORT_ONNX_MODELS or force_export):
        return

    kwargs = {}
    if use_fallback:
        # set the aten fallback to true
        kwargs['operator_export_type'] = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK

    torch.onnx.export(model,
                      x,
                      filepath,
                      export_params=True,
                      input_names=input_names,
                      output_names=output_names,
                      do_constant_folding=True,
                      verbose=False,
                      training=training_mode,
                      opset_version=11,
                      **kwargs)

    return True
