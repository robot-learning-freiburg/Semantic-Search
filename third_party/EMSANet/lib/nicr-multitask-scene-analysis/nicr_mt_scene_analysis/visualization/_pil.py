# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np
from PIL import Image


def to_pil_img(img: np.ndarray, palette=None) -> Image.Image:
    # note that OpenCV is not able to handle indexed pngs correctly.
    if img.max() > 255:
        if palette is not None:
            # we cannot use an indexed image with color palette as it it
            # uint8 only, apply colorization before
            img_colored = np.asarray(palette, dtype='uint8')[img]
            return Image.fromarray(img_colored)
        else:
            return Image.fromarray(img.astype('uint16'))

    # uint8
    img_pil = Image.fromarray(img.astype('uint8'))
    if palette is not None:
        p = list(np.asarray(palette[:256], dtype='uint8').flatten())
        img_pil.putpalette(p)

    return img_pil
