# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""

from ._misc import partial_class

from ._normal import NormalOutputNormalization

from ._orientation import np_deg2biternion
from ._orientation import np_rad2biternion
from ._orientation import np_biternion2deg
from ._orientation import np_biternion2rad
from ._orientation import np_rad2deg

from ._orientation import biternion2deg
from ._orientation import biternion2rad
from ._orientation import OrientationOutputNormalization

from ._printing import cprint
from ._printing import cprint_section
from ._printing import cprint_step
