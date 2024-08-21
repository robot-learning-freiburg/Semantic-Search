# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import os
import sys

from setuptools import find_packages
from setuptools import setup


def run_setup():
    # get version
    version_namespace = {}
    with open(os.path.join('nicr_mt_scene_analysis', 'version.py')) as f:
        exec(f.read(), version_namespace)
    version = version_namespace['_get_version'](with_suffix=False)

    requirements = [
        'matplotlib',
        'numpy',
        'pillow',
        'scipy',
        'nicr_scene_analysis_datasets>=0.5.2'
    ]
    if sys.version_info <= (3, 7):
        requirements.append('torchmetrics==0.8.2')
    else:
        requirements.append('torchmetrics==0.10.2')

    # torch and torchvision should be installed before running this!
    try:
        import torch
    except ImportError:
        raise ImportError(
            "Torch not found. Please install PyTorch before on "
            "your own: https://pytorch.org/get-started/locally/"
        )

    try:
        import torchvision
    except ImportError:
        raise ImportError(
            "TorchVision not found. Please install TorchVision before on "
            "your own: https://pytorch.org/get-started/locally/"
        )

    if sys.version_info <= (3, 7):
        # python 3.6 does not support dataclasses (install backport)
        requirements.append('dataclasses')

    # OpenCV might be installed using another name
    try:
        import cv2
    except ImportError:
        requirements.append('opencv-python')

    # setup
    setup(name='nicr_mt_scene_analysis',
          version='{}.{}.{}'.format(*version),
          description='Python package for multi-task scene analysis',
          author='Daniel Seichter, Söhnke Benedikt Fischedick',
          author_email='daniel.seichter@tu-ilmenau.de, '
                       'soehnke-benedikt.fischedick@tu-ilmenau.de',
          license='Copyright 2021-2023, Neuroinformatics and Cognitive '
                  'Robotics Lab TU Ilmenau, Ilmenau, Germany',
          install_requires=requirements,
          packages=find_packages(),
          extras_require={
              'test': [
                  'scikit-image',
                  'pytest>=3.0.2',
              ]
          },
          include_package_data=True,
          package_data={
              'nicr_mt_scene_analysis': [
                  'visualization/FreeMonoBold.ttf',
              ]
          })


if __name__ == '__main__':
    run_setup()
