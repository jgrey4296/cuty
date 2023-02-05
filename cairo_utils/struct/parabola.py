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

@dataclass
class ParabolaVertex:
    """
    Vertex form of a parabola:
    y = a(x-h)^2 + k
    """
    a : float = field(default=0)
    h : float = field(default=0)
    k : float = field(default=0)

    def __call__(self, x):
        return self.a * pow(x + self.h, 2) + self.k

    def to_standard_form(self):
        """ Calculate the standard form of the parabola from a vertex form """
        return StandardForm(self.a, -2*self.a*self.h, a*pow(self.h, 2)+self.k)

@dataclass
class ParabolaStandard:
    """
    standard form of a parabolar:
    y = ax^2 + bx + c
    """
    a : float = field()
    b : float = field()
    c : float = field()

    def __str__(self):
        return "y = {0:.2f} * x^2 + {1:.2f} x + {2:.2f}".format(self.a, self.b, self.c)

    def __call__(self, x):
        return (self.a * pow(x, 2)) + (self.b * x) + self.c

    def to_vertex_form(self):
        """ Calculate the vertex form of the parabola from a standard form """
        return VertexForm(a, -b/(2*a), c-(a * (pow(b, 2) / 4 * a)))

@dataclass
class ParabolaData:
    """
    A class to represent and calculate a parabola. holds both forms of definition,
    focus/directrix AND standard form.
    Handles the degenerate case of focus and directrix being the same by designating
    as a vertical line
    """
    fx            : float = field()
    fy            : float = field()
    d             : float = field()
    vertical_line : bool  = field(default=True)
    p             : float = field(init=False, default=0)

    vert          : VertexForm   = field(init=False)
    stan          : StandardForm = field(init=False)

    id = 0

    def __post_init__(self):
        """ Create a parabola with a focus x and y, and a directrix d """
        #pylint: disable=invalid-name
        self.id            = Parabola.id
        Parabola.id += 1
        #focal parameter: the distance from vertex to focus/directrix
        self.p = 0.5 * (self.fy - self.d)

        va = 0
        if not np.allclose(self.fy, self.d):
            va = 1/(2*(self.fy-self.d))

        self.vert = VertexForm(va, -self.fx, self.fy - self.p)
        self.stan = StandardForm(va, 2 * va * -self.fx, va * (pow(-self.fx, 2)) + self.self.fy - self.p)

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return str(self.stan)

    def __repr__(self):
        return "<Parabola>"

    def __call__(self, x, maximum=MAX):
        """ Shorthand for calculating an the entire parabola """
        if self.vertical_line:
            ys = np.linspace(self.fy, maximum)
            xs = np.repeat(self.fx, len(ys))
            return np.column_stack((xs, ys))
        else:
            return np.column_stack((x, self.calc_standard_form(x)))

    def __eq__(self, parabola2):
        """ Compare two parabolas """
        if not isinstance(parabola2, (type(None), Parabola)):
            raise TypeError()

        if parabola2 is None:
            return False
        a = self.to_numpy_array()
        b = parabola2.to_numpy_array()
        return np.allclose(a, b)
