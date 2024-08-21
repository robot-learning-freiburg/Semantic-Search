# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np
from PIL.Image import Image

from ._pil import to_pil_img


def visualize_normal(normal_img: np.ndarray) -> np.ndarray:
    # assumes for normal: -1 <= normal <= 1, 3 channels, and channels last
    assert normal_img.ndim == 3
    assert normal_img.shape[-1] == 3

    # reverse processing in dataset class
    normal_img = normal_img + 1
    normal_img *= 127

    # clip to uint8 (useful for float32 precision and unit length vectors)
    normal_img = np.clip(normal_img, 0, 255).astype('uint8')

    return normal_img


def visualize_normal_pil(normal_img: np.ndarray) -> Image:
    return to_pil_img(visualize_normal(normal_img), palette=None)
