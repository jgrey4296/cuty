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

def bezier1cp(a_cp_b, t, f=None, p=None):
    """
    Given the start, end, and a control point, create t number of points along that bezier
    t : the number of points to linearly create used to sample along the bezier
    f : a transform function for the sample points prior to calculate bezier
    p : an overriding set of arbitrary sample points for calculate bezier
    """
    #pylint: disable=unused-variable
    assert(isinstance(a_cp_b, np.ndarray))
    if p is not None:
        assert(isinstance(p, np.ndarray))
        sample_points = p
    else:
        sample_points = np.linspace(0, 1, t)
        if f is not None:
            assert(callable(f))
            #f is an easing lookup function
            sample_points = f(t)

    len_samples = len(sample_points)
    quadratic_matrix = np.array([[[1,0,0],[-2,2,0],[1,-2,1]]])
    t_matrix = lambda x: np.array([1, x,  pow(x,2)])

    points = a_cp_b.reshape((-1,3,2))
    results = np.zeros((len(a_cp_b),1,2))
    for t in sample_points:
        t_mul_mat = t_matrix(t) @ quadratic_matrix
        t_lines = t_mul_mat @ points
        results = np.hstack((results, t_lines.reshape(-1,1,2)))
    return results[:,1:,:].reshape((-1,2))

def bezier2cp(a_cpcp_b, n=None, p=None, easing=None):
    """
    Given a start, end, and two control points along the way,
    create n number of points along that bezier
    n : The number of points to sample linearly
    f : the transform function for the linear sampling
    p : arbitrary points to use for sampling instead
    Matrix Multiplication algorithm from:
    https://pomax.github.io/bezierinfo/#decasteljau
    """
    #pylint: disable=too-many-locals
    #pylint: disable=unused-variable
    assert(isinstance(a_cpcp_b, np.ndarray))
    if p is not None:
        assert(isinstance(p, np.ndarray))
        sample_points = p
    elif n is not None:
        sample_points = np.linspace(0, 1, n)
    else:
        raise Exception("Neither arbitrary points or n given")
    if easing is not None:
        assert(callable(easing))
        sample_points = easing(sample_points)

    len_samples = len(sample_points)
    cubic_matrix = np.array([[1,0,0,0],[-3,3,0,0],[3,-6,3,0],[-1,3,-3,1]])
    t_matrix = lambda x: np.array([1, x,  pow(x,2), pow(x,3)])

    points = a_cpcp_b.reshape((-1,4,2))
    results = np.zeros((len(a_cpcp_b),1,2))
    for t in sample_points:
        t_mul_mat = t_matrix(t) @ cubic_matrix
        t_lines = t_mul_mat @ points
        results = np.hstack((results, t_lines.reshape(-1,1,2)))
    return results[:,1:,:].reshape((-1,2))


def is_point_on_line(p, l):
    """ Test to see if a point is on a line """
    assert(isinstance(p, np.ndarray))
    assert(isinstance(l, np.ndarray))
    points = p.reshape((-1, 2))
    the_lines = l.reshape((-1, 2, 2))
    l_mins = the_lines.min(axis=1)
    l_maxs = the_lines.max(axis=1)

    in_bounds_xs = l_mins[:, 0] <= points[:, 0] <= l_maxs[:, 0]
    in_bounds_ys = l_mins[:, 1] <= points[:, 1] <= l_maxs[:, 1]

    if np.allclose((the_lines[:, 0, 0] - the_lines[:, 1, 0]), 0):
        return in_bounds_ys and in_bounds_xs
    slopes = (the_lines[:, 0, 1] - the_lines[:, 1, 1]) / (the_lines[:, 0, 0] - the_lines[:, 1, 0])
    y_intersects = - slopes * the_lines[:, 0, 0] + the_lines[:, 0, 1]
    line_ys = slopes * points[:, 0] + y_intersects
    return np.allclose(line_ys, points[0, 1]) and in_bounds_ys and in_bounds_xs

def make_horizontal_lines(n=1, random=None):
    """ Utility to Describe a horizontal line as a vector of start and end points  """
    if random is None:
        random = np.random.random
    x = random((n, 2)).sort()
    y = random(n).reshape((-1, 1))
    return np.column_stack((x[:, 0], y, x[:, 1], y))

def make_vertical_lines(n=1, random=None):
    """ utility Describe a vertical line as a vector of start and end points """
    if random is None:
        random =np.random.random
    x = random(n).reshape((-1, 1))
    y = random((n, 2)).sort()
    return np.column_stack((x, y[:, 0], x, y[:, 1]))

def sample_along_lines(xys, n, easing=None, override=None):
    """ For a set of lines, sample along them n times,
    with easings distribution. can be hard overriden with override
    """
    t = np.linspace(0,1,n)
    if override is not None:
        t = override
    if easing is not None:
        t = easing(t)
    s_points = t
    s_invert = (1 - s_points)
    num_points = s_points.shape[0]
    fade = np.vstack((s_invert,s_points))
    lines = xys.reshape((-1, 2, 2))
    xs = lines[:, :, 0]
    ys = lines[:, :, 1]
    xsr = xs.repeat(num_points).reshape((-1, 2, num_points))
    ysr = ys.repeat(num_points).reshape((-1, 2, num_points))
    xsrf = xsr * fade
    ysrf = ysr * fade
    s_xs = xsrf.sum(axis=1)
    s_ys = ysrf.sum(axis=1)
    paired = np.hstack((s_xs, s_ys))
    reshaped = paired.reshape((-1, num_points, 2), order='F').reshape((-1,2))
    return reshaped


def create_line(xys, n, easing=None):
    """ Given a start and end, create t number of points along that line """
    line = sample_along_lines(xys.reshape((-1,LINE_DATA_LEN)), n, easing=easing)
    return line

def get_bisector(p1, p2, r=False):
    """ With a normalised line, rotate 90 degrees,
    r=True : to the right
    r=False : to the left
    """
    n = get_unit_vector(p1, p2)
    if r:
        n_prime = n.dot([[0, -1],
                         [1, 0]])
    else:
        n_prime = n.dot([[0, 1],
                         [-1, 0]])
    return n_prime



def intersect(line_1, line_2, tolerance=TOLERANCE):
    """ Get the intersection points of two line segments
    see: http://ericleong.me/research/circle-line/
    so line_1:(start, end), line_2:(start, end)
    returns np.array([x, y]) of intersection or None
    """
    #pylint: disable=too-many-locals
    assert(isinstance(line_1, np.ndarray))
    assert(isinstance(line_2, np.ndarray))
    assert(line_1.shape == (2, 2))
    assert(line_2.shape == (2, 2))
    #The points
    p0 = line_1[0]
    p1 = line_1[1]
    p2 = line_2[0]
    p3 = line_2[1]

    a1 = p1[1] - p0[1]
    b1 = p0[0] - p1[0]
    c1b = a1 * p0[0] + b1 * p0[1]

    a2 = p3[1] - p2[1]
    b2 = p2[0] - p3[0]
    c2b = a2*p2[0] + b2*p2[1]

    detb = a1 * b2 - a2 * b1
    if detb == 0:
        return None

    xb = ((c1b * b2) - (b1 * c2b)) / detb
    yb = ((a1 * c2b) - (c1b * a2)) / detb
    xyb = np.array([xb, yb])

    l1mins = np.min((p0, p1), axis=0) - tolerance
    l2mins = np.min((p2, p3), axis=0) - tolerance
    l1maxs = np.max((p0, p1), axis=0) + tolerance
    l2maxs = np.max((p2, p3), axis=0) + tolerance

    if (l1mins <= xyb).all() and (l2mins <= xyb).all() and \
       (xyb <= l1maxs).all() and (xyb <= l2maxs).all():
        return xyb
    return None


def extend_line(p1, p2, m=1, from_start=True):
    """ Extend a line by m units
    Returns the new end points only
    """
    n = get_unit_vector(p1, p2)
    if from_start:
        el = p1 + (n * m)
    else:
        el = p2 + (n * m)
    return el
