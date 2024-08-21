# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Union

from .dwa import DynamicWeightAverage
from .rlw import RandomLossWeighting
from .fixed import FixedLossWeighting

LossWeightingType = Union[DynamicWeightAverage, RandomLossWeighting,
                          FixedLossWeighting]
