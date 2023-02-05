#!/usr/bin/env python3
"""

"""
##-- imports
from __future__ import annotations

import types
import abc
import datetime
import enum
import functools as ftz
import itertools as itz
import logging as logmod
import pathlib as pl
import re
import time
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Final, Generic,
                    Iterable, Iterator, Mapping, Match, MutableMapping,
                    Protocol, Sequence, Tuple, TypeAlias, TypeGuard, TypeVar,
                    cast, final, overload, runtime_checkable)
from uuid import UUID, uuid1
from weakref import ref

##-- end imports

##-- logging
logging = logmod.getLogger(__name__)
##-- end logging

class FuncLookup:
    pass

ELOOKUP = {
    "pow_abs_curve"  : pow_abs_curve,
    "pow_cos_pi"     : pow_cos_pi,
    "pow_abs_sin_pi" : pow_abs_sin_pi,
    "pow_min_cos_pi" : pow_min_cos_pi,
    "pow_max_abs"    : pow_max_abs,
    "sigmoid"        : sigmoid,
    "linear"         : linear,
    "static"         : static,
    "soft_knee"      : soft_knee
}
ENAMES = list(ELOOKUP.keys())

def lookup(name):
    """ Lookup a curve function using a given name """
    if isinstance(name, int):
        return ELOOKUP[ENAMES[name]]
    elif name not in ELOOKUP:
        raise Exception("Unrecognised easing name: {}".format(name))
    return ELOOKUP[name]
