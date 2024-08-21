# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional, Tuple

import inspect

import torch.nn as nn

from ..utils import cprint
from .block import BasicBlock
from .block import Bottleneck
from .block import NonBottleneck1D


def get_class_name(cls_or_obj):
    # for some network parts, we use a class factory to pass parameters,
    # get the name of the first real class in that case
    cls = cls_or_obj if inspect.isclass(cls_or_obj) else cls_or_obj.__class__

    for c in inspect.getmro(cls):
        if 'PartialClass' == c.__name__:
            continue
        return c.__name__


def he_initialization(
    module: nn.Module,
    blacklist: Tuple[nn.Module] = (),
    name_hint: Optional[str] = None,
    debug: bool = True,
) -> None:
    # according to master thesis of C. Kaestner, all convs following a ReLU
    # should be he-initialized

    def _he_initialization(module, name_hint=None, recursion_hint=0):
        def dprint(txt_fmt, cur_name, cur_module, *, color=None):
            if not debug:
                return
            name_str = f'{name_hint or get_class_name(module)}'
            name_str += f'.{cur_name} ({get_class_name(cur_module)})'
            recursion_hint_str = '\t'*recursion_hint + '└─ '
            cprint(recursion_hint_str + txt_fmt.format(name_str), color=color)

        for n, m in module.named_children():
            if blacklist and isinstance(m, blacklist):
                dprint("Module '{}' skipped.", n, m, color='grey')
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                dprint("Applied he init to module '{}'", n, m, color='green')
                continue

            dprint("Entering '{}'", n, m)
            _he_initialization(m, recursion_hint=recursion_hint+1)

    # apply initialization
    class_name = get_class_name(module)
    name_str = f'{name_hint} ({class_name})' if name_hint else f'{class_name}'
    if debug:
        cprint(f"Initializing weights in '{name_str}'", color='green')
    _he_initialization(module, name_hint=name_hint)


def zero_residual_initialization(module: nn.Module, debug: bool = True) -> None:
    for n, m in module.named_modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.norm3.weight, 0)
        elif isinstance(m, (BasicBlock, NonBottleneck1D)):
            nn.init.constant_(m.norm2.weight, 0)
        else:
            continue

        if debug:
            cprint(f"Applied zero-residual initialization to '{n} "
                   f"({get_class_name(m)})'",
                   color='green')
