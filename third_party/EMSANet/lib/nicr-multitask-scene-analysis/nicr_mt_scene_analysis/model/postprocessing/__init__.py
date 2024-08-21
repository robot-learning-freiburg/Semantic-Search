# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Type, Union

from ...utils import partial_class
from .instance import InstancePostprocessing
from .normal import NormalPostprocessing
from .panoptic import PanopticPostprocessing
from .scene import ScenePostprocessing
from .semantic import SemanticPostprocessing

PostProcessingType = Type[
    Union[
        InstancePostprocessing,
        NormalPostprocessing,
        PanopticPostprocessing,
        ScenePostprocessing,
        SemanticPostprocessing,
    ]
]


def get_postprocessing_class(name: str, **kwargs: Any) -> Type[PostProcessingType]:
    if "semantic" == name:
        cls = SemanticPostprocessing
    elif "instance" == name:
        cls = InstancePostprocessing
    elif "panoptic" == name:
        cls = PanopticPostprocessing
    elif "normal" == name:
        cls = NormalPostprocessing
    elif "scene" == name:
        cls = ScenePostprocessing
    else:
        raise ValueError(f"Unknown postprocessing: '{name}'")
    return partial_class(cls, **kwargs)
