# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Sequence, Tuple, Union

import numpy as np
from PIL.Image import Image

from ._pil import to_pil_img


def visualize_semantic(
    semantic_img: np.array,
    colors: Union[Sequence[Tuple[int, int, int]], np.ndarray]
) -> np.array:
    assert semantic_img.ndim == 2
    cmap = np.asarray(colors, dtype='uint8')
    return cmap[semantic_img]


def visualize_semantic_pil(
    semantic_img: np.array,
    colors: Union[Sequence[Tuple[int, int, int]], np.ndarray]
) -> Image:
    assert semantic_img.ndim == 2
    return to_pil_img(semantic_img, palette=colors)
