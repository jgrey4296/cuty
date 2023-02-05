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

def check_distance_from_points(point, quad_tree, dist=HALFDELTA):
    """ Given a point and a quadtree, return true if the point is within
    the bounds of the quadtree but not near any points """
    bbox = [point[0] - dist, point[1] - dist, point[0] + dist, point[1] + dist]
    area = quad_tree.intersect(bbox)
    inside_canvas = point[0] > 0 and point[0] < 1.0 and point[1] > 0 and point[1] < 1.0
    return (not bool(area)) and inside_canvas

def get_closest_to_focus(focus, possible_points):
    """ Given a set of points, return the point closest to the focus """
    ds = get_distance(focus, possible_points)
    m_d = ds.min()
    i = ds.tolist().index(m_d)
    return possible_points[i]

def get_closest_on_side(ref_point, possible_points, left=True):
    """
    given a reference point and a set of candidates, get the closest
    point on either the left or right of that reference
    """
    subbed = possible_points - ref_point
    if left:
        on_side = subbed[:, 0] < 0
    else:
        on_side = subbed[:, 0] > 0
    try:
        i = on_side.tolist().index(True)
        return possible_points[i]
    except ValueError:
        return None

def get_distance_raw(p1, p2):
    """ Get the non-square-root distance for pairs of points """
    assert(isinstance(p1, np.ndarray))
    assert(isinstance(p2, np.ndarray))
    p1 = p1.reshape(-1, 2)
    p2 = p2.reshape(-1, 2)
    d_squared = pow(p2-p1, 2)
    #summed = dSquared[:, 0] + dSquared[:, 1]
    summed = d_squared.sum(axis=1)
    return summed

def get_distance(p1, p2):
    """ Get the square-root distance of pairs of points """
    assert(isinstance(p1, np.ndarray))
    assert(isinstance(p2, np.ndarray))

    summed = get_distance_raw(p1, p2)
    sqrtd = np.sqrt(summed)
    return sqrtd

def get_distance_xyxy(x1, y1, x2, y2):
    """ Utility to get the raw distance of points as separate x's and y's  """
    return get_distance_raw(np.array([x1, y1]), np.array([x2, y2]))[0]

def get_midpoint(p1, p2):
    """ Given two points, get the point directly between them """
    m = (p1 + p2) / 2
    return m

def radians_between_points(a, b):
    """
    takes np.arrays
    return the radian relation of b to a (source)
    ie: if > 0: anti-clockwise, < 0: clockwise
    """
    c = b - a
    return atan2(c[1], c[0])

