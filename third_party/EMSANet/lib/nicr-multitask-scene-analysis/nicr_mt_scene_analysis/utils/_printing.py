# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""

try:
    import termcolor
    cprint = termcolor.cprint

except ImportError:

    def cprint(*args, **kwargs):
        kwargs.pop('color', None)
        kwargs.pop('on_color', None)
        kwargs.pop('attrs', None)

        print(*args, **kwargs)


def cprint_section(*args, **kwargs):
    cprint('-'*40, **kwargs)
    cprint(*args, **kwargs)
    cprint('-'*40, **kwargs)


def cprint_step(*args):
    cprint(*args, color='blue', attrs=('bold',))
