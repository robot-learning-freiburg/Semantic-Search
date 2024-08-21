# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np
from skimage import data
import torch

from ..data.preprocessing.utils import _get_relevant_spatial_keys
from ..visualization import visualize_instance_offset
from ..visualization import visualize_orientation
from . import SHOW_RESULTS


def show_results(sample, sample_pre, title, force_show=False):
    if not (SHOW_RESULTS or force_show):
        return

    import matplotlib.pyplot as plt

    # flat dict for multiscale supersion
    for key in list(sample_pre.keys()):
        if '_down' in key:
            for k, v in sample_pre[key].items():
                sample_pre[f'{key}->{k}'] = v

    # flat dict for cloned inputs
    for key in list(sample_pre.keys()):
        if '_no_preprocessing' in key:
            for k, v in sample_pre[key].items():
                sample_pre[f'{key}->{k}'] = v

    # check for torch stuff
    for key, value in sample_pre.items():
        if isinstance(value, torch.Tensor):
            value = value.numpy()
            if value.ndim == 3:
                # channels last
                value = value.transpose((1, 2, 0))
            sample_pre[key] = value

    # the preprocessed sample may contain more keys
    keys = _get_relevant_spatial_keys(sample_pre)

    fig, axes = plt.subplots(2, len(keys))
    for i, key in enumerate(keys):
        axes_key = axes[0, i], axes[1, i]

        # get preprocessed image
        img_pre = sample_pre[key]

        # get original image or empty image if key does not exist
        if key in sample:
            img = sample[key]
        else:
            img = np.zeros(img_pre.shape[:2])

        if 'rgb' in key:
            axes_key[0].imshow(img, interpolation='nearest')
            axes_key[1].imshow(img_pre, interpolation='nearest')
        elif 'depth' in key:
            axes_key[0].imshow(img,
                               vmin=img.min(), vmax=img.max(),
                               cmap='gray', interpolation='nearest')
            axes_key[1].imshow(img_pre,
                               vmin=img_pre.min(), vmax=img_pre.max(),
                               cmap='gray', interpolation='nearest')
        elif 'instance_offset' in key:
            axes_key[0].imshow(img,
                               cmap='gray', interpolation='nearest')
            axes_key[1].imshow(visualize_instance_offset(img_pre),
                               interpolation='nearest')
        elif ('orientation' in key) and ('foreground' not in key):
            axes_key[0].imshow(img,
                               cmap='gray', interpolation='nearest')
            axes_key[1].imshow(visualize_orientation(img_pre),
                               interpolation='nearest')
        else:
            axes_key[0].imshow(img,
                               cmap='gray', interpolation='nearest')
            axes_key[1].imshow(img_pre,
                               cmap='gray', interpolation='nearest')

        axes_key[0].set_title(key, fontsize=8)
        axes_key[1].set_title(key, fontsize=8)

        h, w = img.shape[:2]
        h_pre, w_pre = img_pre.shape[:2]

        axes_key[0].set_yticks([0, h])
        axes_key[0].set_xticks([0, w])
        axes_key[1].set_yticks([0, h_pre])
        axes_key[1].set_xticks([0, w_pre])
        axes_key[0].tick_params(axis='both', labelsize=6)
        axes_key[1].tick_params(axis='both', labelsize=6)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def get_dummy_sample():
    img = data.astronaut()
    # get dummy depth as first channel * 64
    depth = img[..., 0].copy().astype('uint16') * 2**6
    # get some dummy mask by thresholding the second channel
    some_mask = (img[..., 1] > 100)

    # create some instance image and semantic segmentation
    shape = depth.shape
    instance = np.zeros(shape, dtype='uint8')
    instance[40:100, 40:100] = 1
    instance[150:200, 40:100] = 2
    instance[0:200, 200:250] = 3
    instance[300:400, 200:250] = 4
    semantic = np.zeros(shape, dtype='uint8')
    semantic[40:100, 40:100] = 1
    semantic[150:200, 40:100] = 2
    semantic[0:200, 200:250] = 10    # not relevant for instances/orientation
    semantic[300:400, 200:250] = 3

    # assign some orientations
    # we have orientations for instance 1, 2, and 3
    orientations = {1: np.deg2rad(45),
                    2: np.deg2rad(90),
                    3: np.deg2rad(135)}

    return {'rgb': img,
            'depth': depth,
            'instance': instance,
            'semantic': semantic,
            'some_mask': some_mask,
            'orientations': orientations,
            'scene': 0}
