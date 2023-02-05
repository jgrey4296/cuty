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

def sample_circle(xyrs, n, sort_rads=True, sort_radi=True, easing=None, random=None):
    """
    https://stackoverflow.com/questions/5837572
    """
    if random is None:
        random = np.random.random

    xyrs_r = xyrs.reshape((-1, 1, CIRCLE_DATA_LEN)).repeat(n, axis=1)

    r = scale_ndarray(np.sqrt(random((n, 1))), xyrs_r[0,:,4:])
    if sort_radi:
        r.sort(axis=1)

    theta = scale_ndarray(random((n, 1)), xyrs_r[0,:,2:4])
    if sort_rads:
        theta.sort(axis=1)

    r_shaped = r.reshape((xyrs_r.shape[0],n,1))
    theta_shaped = theta.reshape((xyrs_r.shape[0],n,1))

    rot = np.dstack((np.cos(theta_shaped), np.sin(theta_shaped)))
    r_x_rot = r_shaped * rot

    result = xyrs_r[:,:,:2] + r_x_rot
    flattened = result.reshape((-1, 2))
    return flattened

def get_circle_3p(p1, p2, p3, arb_intersect=20000):
    """
    Given 3 points, treat them as defining two chords on a circle,
    intersect them to find the centre, then calculate the radius
    Thus: circumcircle
    """
    #pylint: disable=too-many-locals
    assert(all([isinstance(x, np.ndarray) for x in [p1, p2, p3]]))
    sorted_points = sort_coords(np.array([p1, p2, p3]))
    p1 = sorted_points[0]
    p2 = sorted_points[1]
    p3 = sorted_points[2]

    arb_height = arb_intersect
    #mid points and norms:
    m1 = get_midpoint(p1, p2)
    n1 = get_bisector(m1, p2)
    m2 = get_midpoint(p2, p3)
    n2 = get_bisector(m2, p3)
    #extended norms:
    v1 = m1 + (1 * arb_height * n1)
    v2 = m2 + (1 * arb_height * n2)
    v1_i = m1 + (-1 * arb_height * n1)
    v2_i = m2 + (-1 * arb_height * n2)
    #resulting lines:
    l1 = np.row_stack((m1, v1))
    l2 = np.row_stack((m2, v2))
    l1_i = np.row_stack((m1, v1_i))
    l2_i = np.row_stack((m2, v2_i))
    #intersect extended norms:
    #in the four combinations of directions
    i_1 = intersect(l1, l2)
    i_2 = intersect(l1_i, l2_i)
    i_3 = intersect(l1, l2_i)
    i_4 = intersect(l1_i, l2)
    #get the intersection:
    the_intersect = [x for x in [i_1, i_2, i_3, i_4] if x is not None]
    if the_intersect is None or not bool(the_intersect):
        return None
    r1 = get_distance(p1, the_intersect[0])
    r2 = get_distance(p2, the_intersect[0])
    r3 = get_distance(p3, the_intersect[0])

    #a circle only if they are have the same radius
    if np.isclose(r1, r2) and np.isclose(r2, r3):
        return (the_intersect[0], r1)
    else:
        return None

def get_lowest_point_on_circle(centre, radius):
    """
    given the centre of a circle and a radius, get the lowest y point on that circle
    """
    return centre - np.array([0, radius])

def in_circle(centre, radius, points):
    """ Test a set of points to see if they are within a circle's radius """
    d = get_distance_raw(centre, points)
    return d < pow(radius, 2)
