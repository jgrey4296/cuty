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

def granulate(xys, grains=10, mult=2, random=None):
    """ Given a set of points, offset each slightly
    by the direction between the points
    """
    assert(isinstance(xys, np.ndarray))
    assert(len(xys.shape) == 2)
    if random is None:
        random = np.random.random
    directions, hypos = get_directions(xys)
    granulated = np.zeros((1, 2))
    for i, d in enumerate(hypos):
        sub_granules = xys[i, :] + (d * directions[i, :]*(random((grains, 1))) * mult)
        granulated = np.row_stack((granulated, sub_granules))
    return granulated[1:]

def vary(xys, step_size, pix, random=None):
    """
    FIXME : investigate
    for a given set of points, wiggle them slightly
    """
    assert(isinstance(xys, np.ndarray))
    assert(len(xys.shape) == 2)
    if random is None:
        random = np.random.random
    r = (1.0-2.0 * random((len(xys), 1)))
    scale = np.reshape(np.arange(len(xys)).astype('float'), (len(xys), 1))
    noise = (r* scale * step_size)
    a = random(len(xys))
    rnd = np.column_stack((np.cos(a), np.sin(a)))
    rnd_noise = rnd * noise
    rnd_noise_pix = rnd_noise * pix
    xys_prime = xys + rnd_noise_pix
    return xys_prime


def sample_wrapper(func, data, n, radius, colour, easing=None, random=None):
    """ A Wrapper for sample_circle, sample_along_lines,
    bezier1cp and bezier2cp
    Takes the function, applies the data and samp_sig_or_count to it,
    then duplicates the radius and colours to the appropriate count,
    and combines
    """
    assert(callable(func))
    sampled = func(data, n, easing=easing, random=random)
    radius_size = len(data) * n
    repeated_radius = np.repeat(radius, radius_size).reshape((radius_size, 1))
    repeated_colours = np.repeat(colour.reshape((-1,4)), len(sampled), axis=0)
    combined = np.column_stack((sampled, repeated_radius, repeated_colours))
    return combined


def displace_around_circle(xys, scale, n, random=None):
    """ displace the data around a scaled noisy circle """
    #pylint: disable=invalid-name
    #Create a circle:
    if random is None:
        random = np.random.random
    t = np.linspace(0, 2*pi, n)
    rotation = np.column_stack((sin(t), cos(t))).transpose()
    #create some noise:
    noise = random(n)
    #jitter the rotation:
    jittered = (rotation * noise)
    #control the amount of this noise to apply
    scaled = (jittered * scale).transpose()
    #apply the noise to the data
    mod_points = xys + scaled
    return mod_points


def calculate_single_point(points, d=DELTA, random=None):
    """ points passed in, move in a random direction """
    if random is None:
        random = np.random.random
    arr = random((points.shape[0], 2)) * TWOPI
    delta = np.array([sin(arr[:, 0]), cos(arr[:, 1])]) * (2 * d)
    return points + delta

def calculate_vector_point(ps, d=DELTA, random=None):
    """ passed in pairs of points, move in the direction of the vector """
    if random is None:
        random = np.random.random
    vector = ps[:, 2:] - ps[:, :2]
    rand_amnt = random((ps.shape[0], 2)) * TWOPI
    mag = np.sqrt(np.sum(np.square(vector)))
    norm_vector = vector / mag
    move_vector = norm_vector * (2 * d)
    jiggled_vector = move_vector * np.array([sin(rand_amnt[:, 0]), cos(rand_amnt[:, 1])])
    return ps[:, 2:] + jiggled_vector
