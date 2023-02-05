"""
Math Utilities
TODO: Refactor into submodules
"""
##-- imports
from __future__ import annotations

import logging as root_logger
from functools import partial
from math import atan2, copysign
import numpy as np
from numpy import cos, sin, pi
from scipy.interpolate import splprep, splev

from ..constants import PI, TWOPI, QUARTERPI, EPSILON, TOLERANCE
from ..constants import IntersectEnum, DELTA, HALFDELTA, NODE_RECIPROCAL
from ..constants import SAMPLE_DATA_LEN, LINE_DATA_LEN, BEZIER_DATA_LEN, CIRCLE_DATA_LEN

##-- end imports

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

logging = root_logger.getLogger(__name__)

def _interpolate(xy, num_points, smoothing=0.2):
    """ given a set of points, generate values between those points """
    assert(isinstance(xy, np.ndarray))
    assert(len(xy.shape) == 2)
    assert(xy.shape[0] >= 4)
    assert(xy.shape[1] == 2)
    spline_tuple, _ = splprep([xy[:, 0], xy[:, 1]], s=smoothing)
    interpolate_points = np.linspace(0, 1, num_points)
    smoothed_xy = np.column_stack(splev(interpolate_points, spline_tuple))
    return smoothed_xy

def scale(xs, minmax):
    result = minmax[0] + (xs * (minmax[1] - minmax[0]))
    return result

def scale_ndarray(xs, minmaxs):
    mins = minmaxs[:,0].reshape((-1,1))
    ranges = (minmaxs[:,1] - minmaxs[:,0]).reshape((-1,1))
    result = mins + (xs * ranges)
    return result

def check_sign(a, b):
    """ Test whether two numbers have the same sign """
    return copysign(a, b) == a

def get_min_range_pair(p1, p2):
    """ TODO: Can't remember, test this """
    d1 = get_distance(p1, p2)
    fp2 = np.flipud(p2)
    d2 = get_distance(p1, fp2)
    d1_min = d1.min()
    d2_min = d2.min()
    if d1_min < d2_min:
        i = d1.tolist().index(d1_min)
        #get the right xs
        return np.array([p1[i][0], p2[i][0]])
    else:
        i = d2.tolist().index(d2_min)
        return np.array([p1[i][0], fp2[i][0]])

def clamp(n, minn=0, maxn=1):
    """ Clamp a number between min and max,
    could be replaced with np.clip
    """
    return max(min(maxn, n), minn)

def get_ranges(a):
    """ Given pairs, get the ranges of them """
    assert(isinstance(a, np.ndarray))
    assert(a.shape[1] == 2)
    ranges = np.array([a.min(axis=0), a.max(axis=0)])
    return ranges.T

def node_to_position(x, y):
    """ Convert a nodes XY Position to its actual position """
    return [NODE_RECIPROCAL * x, NODE_RECIPROCAL * y]
