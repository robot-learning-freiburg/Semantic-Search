# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import numpy as np
from PIL.Image import Image

from ._pil import to_pil_img


def visualize_depth(depth_img: np.ndarray) -> np.ndarray:
    assert depth_img.ndim == 2
    _min = depth_img.min()
    _max = depth_img.max()

    # Normalize to [0, 1]
    depth_img = np.float32(depth_img)
    depth_img = (depth_img - _min) / (_max - _min)
    depth_img *= 255
    depth_img = np.uint8(depth_img)
    return depth_img


def visualize_depth_pil(
    depth_img: np.ndarray,
) -> Image:
    assert depth_img.ndim == 2
    return to_pil_img(visualize_depth(depth_img))
