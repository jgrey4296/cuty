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

def random_radian(min_v=-TWOPI, max_v=TWOPI, shape=(1,), random=None):
    """ Get a random value within the range of radians -2pi -> 2pi """
    if random is None:
        random = np.random.random
    return min_v + (random(shape) * (max_v-min_v))

def random_points(n, random=None):
    """ utility to get n 2d points """
    if random is None:
        random = np.random.random
    return random(n*2)

