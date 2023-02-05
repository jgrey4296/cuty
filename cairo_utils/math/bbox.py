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

def bbox_to_lines(bbox, epsilon=EPSILON):
    """ take in the min and max values of a bbox,
    return back a list of 4 lines with the enum designating their position """
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4, ))
    bbox_e = bbox + np.array([-epsilon, -epsilon, epsilon, epsilon])
    # [[minx, miny], [maxx, maxy]] -> [[minx, maxx], [miny, maxy]]
    bbox_t = bbox_e.reshape((2, 2)).transpose()
    #convert the bbox to bounding lines
    select_x = np.array([1, 0])
    select_y = np.array([0, 1])
    mins = bbox_t[:, 0]
    maxs = bbox_t[:, 1]
    min_x_max_y = mins * select_x + maxs * select_y
    max_x_min_y = maxs * select_x + mins * select_y
    lines = [(np.row_stack((mins, max_x_min_y)), IntersectEnum.HBOTTOM),
             (np.row_stack((min_x_max_y, maxs)), IntersectEnum.HTOP),
             (np.row_stack((mins, min_x_max_y)), IntersectEnum.VLEFT),
             (np.row_stack((max_x_min_y, maxs)), IntersectEnum.VRIGHT)]

    return lines

def bound_line_in_bbox(line, bbox):
    """ takes in a line, limits it to be within a bbox """
    #replace original line endpoint with intersection point
    bbl = bbox_to_lines(bbox)
    intersections = [x for x in [intersect(line, x) for x, y in bbl] if x is not None]
    if not bool(intersections):
        return [line]
    return [np.array([line[0], x]) for x in intersections]

def calc_bbox_corner(bbox, ies, epsilon=EPSILON):
    """ Calculate the nearest corner of a bbox for set of existing intersections  """
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4, ))
    assert(isinstance(ies, set))
    hb = IntersectEnum.HBOTTOM
    ht = IntersectEnum.HTOP
    vl = IntersectEnum.VLEFT
    vr = IntersectEnum.VRIGHT
    bbox_e = bbox + np.array([-epsilon, -epsilon, epsilon, epsilon])
    # [[minx, miny], [maxx, maxy]] -> [[minx, maxx], [miny, maxy]]
    bbox_t = bbox_e.reshape((2, 2)).transpose()
    #convert the bbox to bounding lines
    select_x = np.array([1, 0])
    select_y = np.array([0, 1])
    mins = bbox_t[:, 0]
    maxs = bbox_t[:, 1]
    min_x_max_y = mins * select_x + maxs * select_y
    max_x_min_y = maxs * select_x + mins * select_y

    if ies.issubset([hb, vl]):
        return mins
    elif ies.issubset([hb, vr]):
        return max_x_min_y
    elif ies.issubset([ht, vl]):
        return min_x_max_y
    elif ies.issubset([ht, vr]):
        return maxs
    else:
        raise Exception("Calculating box corner failed for: {}".format(ies))

def bbox_centre(bbox):
    """ Get the centre of a bbox """
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4, ))
    bbox_t = bbox.reshape((2, 2)).transpose()
    mins = bbox_t[:, 0]
    maxs = bbox_t[:, 1]
    ranges = maxs - mins
    mid = ranges * 0.5
    return mid

def make_bbox_from_point(point, i):
    """ Given a centre point and a length, create a bbox """
    border = HALFDELTA * (1/(i+1))
    return [point[0] - border, point[1] - border, point[0] + border, point[1] + border]

def within_bbox(point, bbox, tolerance=TOLERANCE):
    """ Test whether a point is within the given bbox """
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4, ))
    assert(isinstance(point, np.ndarray))
    assert(point.shape == (2, ))
    mod_bbox = bbox + np.array([-tolerance, -tolerance, tolerance, tolerance])
    in_x_bounds = mod_bbox[0] < point[0] and point[0] < mod_bbox[2]
    in_y_bounds = mod_bbox[1] < point[1] and point[1] < mod_bbox[3]
    return in_x_bounds and in_y_bounds
