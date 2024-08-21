# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ._pil import to_pil_img


def visualize_heatmap(
    heatmap_img: np.array,
    min_: Optional[int] = None,
    max_: Optional[int] = None,
    cmap: str = 'jet',
) -> np.array:
    min_ = min_ or 0.
    max_ = max_ or 1.

    # normalize heatmap
    heatmap_img_01 = (heatmap_img - min_) / (max_ - min_)

    # apply colormap
    img = plt.get_cmap(cmap)(heatmap_img_01)[..., :3]    # remove alpha channel
    img = (img * 255).astype(np.uint8)

    return img


def visualize_heatmap_pil(
    heatmap_img: np.array,
    min_: Optional[int] = None,
    max_: Optional[int] = None,
    cmap: str = 'jet',
) -> Image.Image:
    return to_pil_img(
        visualize_heatmap(heatmap_img, min_, max_, cmap),
        palette=None
    )
