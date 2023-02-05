#!/usr/bin/env python3
"""

"""
##-- imports
from __future__ import annotations

import abc
import datetime
import enum
import functools as ftz
import itertools as itz
import logging as logmod
import pathlib as pl
import re
import time
import types
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

from functools import partial
from math import atan2, copysign

import cairo_utils as cu
import numpy as np
from cairo_utils.constants import (BEZIER_DATA_LEN, CIRCLE_DATA_LEN, DELTA,
                                   EPSILON, HALFDELTA, LINE_DATA_LEN,
                                   NODE_RECIPROCAL, PI, QUARTERPI,
                                   SAMPLE_DATA_LEN, TOLERANCE, TWOPI,
                                   IntersectEnum)
from numpy import cos, pi, sin
from scipy.interpolate import splev, splprep


def sort_coords(arr):
    """ Sort a list of points by x then y value  """
    ind = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[ind]
