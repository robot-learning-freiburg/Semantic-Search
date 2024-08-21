# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np
import torch

from ._torch import unit_length


def np_rad2deg(rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(rad) % 360


def np_deg2biternion(deg: np.ndarray) -> np.ndarray:
    y = np.deg2rad(deg)
    return np_rad2biternion(y)


def np_rad2biternion(rad: np.ndarray) -> np.ndarray:
    return np.array([np.cos(rad), np.sin(rad)], dtype="float32")


def np_biternion2rad(biternion: np.array) -> np.array:
    # y <- sin, x <- cos
    rad = np.arctan2(biternion[:, 1], biternion[:, 0])
    return rad


def np_biternion2deg(biternion: np.array) -> np.array:
    rad = np_biternion2rad(biternion)
    return np_rad2deg(rad)


def _rad2deg(rad: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(rad) % 360


def biternion2rad(biternion: torch.Tensor) -> torch.Tensor:
    # y <- sin, x <- cos
    rad = torch.atan2(biternion[:, 1], biternion[:, 0])
    return rad


def biternion2deg(biternion: torch.Tensor) -> torch.Tensor:
    rad = biternion2rad(biternion)
    return _rad2deg(rad)


class OrientationOutputNormalization(torch.nn.Module):
    def __init__(self, epsilon=1e-7) -> None:
        self._epsilon = epsilon
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize to: sin(phi)**2 + cos(phi)**2 = 1 (unit circle)
        return unit_length(x, self._epsilon)
