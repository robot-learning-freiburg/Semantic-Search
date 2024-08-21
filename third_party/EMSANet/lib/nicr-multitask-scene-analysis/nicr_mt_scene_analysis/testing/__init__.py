# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import os

# export onnx models
EXPORT_ONNX_MODELS = os.environ.get('EXPORT_ONNX_MODELS',
                                    default=None) is not None

# show results using matplotlib
SHOW_RESULTS = os.environ.get('SHOW_RESULTS', default=None) is not None
