#!/usr/bin/env python3
"""
Colour Module, provides functions to convert between RGBA and HSLA
Code modifed from:
https://stackoverflow.com/questions/3018313/
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

from typing import TypeAlias
import numpy as np

__all__ = ["Colour", "rgba2hsla", "hsla2rgba"]

Colour : TypeAlias = np.array

def rgba2hsla(rgba:np.array):
    """
    Convert an RGBA colour to HSLA
    expects in range 0-1
    """
    c_max = np.max(rgba[:,:-1], axis=1)
    c_min = np.min(rgba[:,:-1], axis=1)
    delta = c_max - c_min

    lightness = (c_max + c_min) * 0.5
    sat = np.zeros((len(rgba),1))
    non_zero_deltas = delta != 0
    sat_div = (1 - np.abs(2 * lightness - 1))
    correct_for_zeros = ((sat_div == 0) * 1) + sat_div
    to_sat = non_zero_deltas * delta
    sat += (to_sat / correct_for_zeros).reshape((-1,1))

    hsla  = np.apply_along_axis(__to_hsla_row_func, 1, np.column_stack((c_max, delta, rgba, sat, lightness, rgba[:,-1])))
    return hsla

def hsla2rgba(hsla):
    """ Expects 0 <= h <= 360,
    0 <= sla <= 1
    """
    # from https://jsfiddle.net/Lamik/reuk63ay/91
    s = hsla[:,1]
    l = hsla[:,2]
    a = s * np.min((l,1-l), axis=0);

    offset = np.array([0, 8, 4])
    h_div = hsla[:,0] / 30

    ht_r = __to_rgba_calc_hue(h_div, 0, l, a)
    ht_g = __to_rgba_calc_hue(h_div, 8, l, a)
    ht_b = __to_rgba_calc_hue(h_div, 4, l, a)

    return np.column_stack((ht_r, ht_g, ht_b, hsla[:,-1]))

def __to_hsla_row_func(the_slice):
    c_max = the_slice[0]
    delta = the_slice[1]
    if delta == 0:
        delta = 1
    rgb = the_slice[2:6]
    hue = 0
    if c_max == rgb[0]:
        hue = np.mod((rgb[1] - rgb[2]) / delta, 6)
    elif c_max == rgb[1]:
        hue = ((rgb[2] - rgb[0]) / delta) + 2
    elif c_max == rgb[2]:
        hue = ((rgb[0] - rgb[1]) / delta) + 4

    hue *= 60
    return np.array((hue, *the_slice[-3:]))

def __to_rgba_calc_hue(h_div, n, l, a):
    k = ((h_div + n)) % 12
    ks_min = np.min((k-3, 9-k, np.ones(k.shape)), axis=0)
    ks_max = np.max((ks_min, -np.ones(k.shape)), axis=0)
    h = l - a * ks_max
    return h
