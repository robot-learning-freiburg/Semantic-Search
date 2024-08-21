# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""

_VERSION_MAJOR = 0
_VERSION_MINOR = 2
_VERSION_MICRO = 1
_VERSION_SUFFIX = ""


def _get_version(with_suffix=False):    # pragma no cover
    """Private function to determine version"""
    if with_suffix:
        return (_VERSION_MAJOR,
                _VERSION_MINOR,
                _VERSION_MICRO,
                _VERSION_SUFFIX)
    else:
        return _VERSION_MAJOR, _VERSION_MINOR, _VERSION_MICRO


__version__ = '{}.{}.{}'.format(*_get_version(with_suffix=False))
