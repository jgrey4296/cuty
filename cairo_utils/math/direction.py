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

def get_random_directions(n=1, random=None):
    """ Choose a direction of cardinal and intercardinal directions """
    dirs = [-1, 0, 1]
    if random is None:
        random = lambda a, x: np.random.choice(a, size=x*2, replace=True, p=None)
    result = random(dirs, n).reshape((n, 2))
    return result

def get_directions(xys):
    """ Given a sequence of points, get the unit direction
    from each point to the next point
    """
    assert(isinstance(xys, np.ndarray))
    assert(len(xys.shape) == 2)
    #convert to vectors:
    #xysPrime.shape = (n, 4)
    #Leading point first to prevent wrap deformation
    # xys_prime = np.column_stack((xys[1:, :], xys[:-1, :]))
    # dx = xys_prime[:, 2] - xys_prime[:, 0]
    # dy = xys_prime[:, 3] - xys_prime[:, 1]
    ds = xys[:-1] - xys[1:]
    #radians:
    arc = np.arctan2(ds[:,1], ds[:,0])
    directions = np.column_stack([np.cos(arc), np.sin(arc)])
    #hypotenuse
    hypos = np.sqrt(np.square(ds[:,0])+np.square(ds[:,1]))
    return np.column_stack((directions, hypos))

def get_unit_vector(p1, p2):
    """ Given two points, get the normalized direction """
    assert(isinstance(p1, np.ndarray))
    assert(isinstance(p2, np.ndarray))
    d = get_distance(p1, p2)
    if np.allclose(d, 0):
        return np.array([0, 0])
    n = (p2-p1)
    normalized = n / d
    return normalized


def is_clockwise(*args, cartesian=True):
    """ Test whether a set of points are in clockwise order  """
    #based on stackoverflow.
    #sum over edges, if positive: CW. negative: CCW
    #assumes normal cartesian of y bottom = 0
    the_sum = 0
    p1s = args
    p2s = list(args[1:])
    p2s.append(args[0])
    pairs = zip(p1s, p2s)
    for p1, p2 in pairs:
        a = (p2[0]-p1[0]) * (p2[1]+p1[1])
        the_sum += a
    if cartesian:
        return the_sum >= 0
    else:
        return the_sum < 0

def is_counter_clockwise(a, b, c):
    """ Given 3 points, do they form a counter clockwise turn """
    assert(all([isinstance(x, np.ndarray) for x in [a, b, c]]))
    offset_b = b - a
    offset_c = c - a
    crossed = np.cross(offset_b, offset_c)
    return crossed >= 0


def __rotate_point_obsolete(p, cen, rads):
    """ Does what rotate point does, explicitly instead
    of with matrix multiplication """
    assert(len(p.shape) == 2)
    c = np.cos(rads)
    s = np.sin(rads)
    centred = p - cen
    cos_p = centred * c
    sin_p = centred * s
    nx = cos_p[:, 0] - sin_p[:, 1]
    ny = sin_p[:, 0] + cos_p[:, 1]
    un_centered = np.column_stack((nx, ny)) + cen
    return un_centered

def rotate_point(p, cen=None, rads=None, rad_min=-QUARTERPI, rad_max=QUARTERPI, random=None):
    """ Given a point, rotate it around a centre point by either radians,
    or within a range of radians
    """
    #p1 = cen, p2=point, @ is matrix mul
    if cen is None:
        cen = np.array([0, 0])
    if rads is None:
        use_radians = random_radian(min_v=rad_min, max_v=rad_max, random=random)
        if isinstance(use_rads, np.ndarray):
            use_radians = use_rads[0]
    else:
        use_radians = rads
    #apply to 1d slices, this allows multiple points to be
    #passed into the function together,
    #without messing up the rotation matmul
    rot_m = rotation_matrix(use_radians)
    offset = (p - cen)
    if len(p.shape) == 1:
        applied = rot_m @ offset
    else:
        applied = np.apply_along_axis(construct_matrix_multiplier(rot_m), 1, offset)
    result = cen + applied
    return result
