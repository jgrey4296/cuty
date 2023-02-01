"""
Provides the Parabola Class, mainly used for Voronoi calculation
"""
import logging as root_logger
from dataclasses import InitVar, dataclass, field
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

import numpy as np

from .quadratic import Quadratic as Q

logging = root_logger.getLogger(__name__)
MAX = 5000

@dataclass
class VertexForm:
    """ Vertex form of a parabola:
    y = a(x-h)^2 + k """
    a : float = field(default=0)
    h : float = field(default=0)
    k : float = field(default=0)

    def __call__(self, x):
        return self.a * pow(x + self.h, 2) + self.k

    def to_standard_form(self):
        """ Calculate the standard form of the parabola from a vertex form """
        return StandardForm(self.a, -2*self.a*self.h, a*pow(self.h, 2)+self.k)

@dataclass
class StandardForm:
    """ standard form of a parabolar:
    y = ax^2 + bx + c """
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
class Parabola:
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

    def is_left_of_focus(self, x):
        """ Given an x, is it smaller than the focus of this parabola """
        return x < self.fx

    def update_d(self, d):
        """ Update the parabola given the directrix has moved """
        self.d = d
        self.p = 0.5 * (self.fy - self.d)
        #Vertex form parameters:
        if np.allclose(self.fy, self.d):
            self.va = 0
            self.vertical_line = True
        else:
            self.va = 1/(2*(self.fy-self.d))
            self.vertical_line = False
        self.vk = self.fy - self.p
        #standard form:
        self.sa = self.va
        self.sb = 2 * self.sa * self.vh
        self.sc = self.sa * (pow(self.vh, 2)) + self.vk

    def intersect(self, p2, d=None):
        """ Take the quadratic representations of parabolas, and
            get the 0, 1 or 2 points that are the intersections
        """
        assert(isinstance(p2, Parabola))
        if d:
            self.update_d(d)
            p2.update_d(d)
        #degenerate cases:
        if self.vertical_line:
            logging.debug("Intersecting vertical line")
            return np.array([p2(self.fx)[0]])
        if p2.vertical_line:
            logging.debug("Intersecting other vertical line")
            return np.array([self(p2.fx)[0]])
        #normal (as quadratic class):
        q1 = Q(self.sa, self.sb, self.sc)
        q2 = Q(p2.sa, p2.sb, p2.sc)
        xs = q1.intersect(q2)
        #logging.debug("Resulting intersects: {}".format(xs))
        xys = self(xs)
        return xys

    def calc_standard_form(self, x):
        """ Get the y value of the parabola at an x position using the standard
            form equation. Should equal calc_vertex_form(x)
        """
        raise DeprecationWarning("use standard form obj")

    def calc_vertex_form(self, x):
        """ Get the y value of the parabola at an x position using
            the vertex form equation. Should equal calc_standard_form(x)
        """
        raise DeprecationWarning("use vertex form obj")

    def calc(self, x):
        """ For given xs, return an (n, 2) array of xy pairs of the parabola """
        return np.column_stack((x, self.calc_vertex_form(x)))

    def __call__(self, x, maximum=MAX):
        """ Shorthand for calculating an the entire parabola """
        if self.vertical_line:
            ys = np.linspace(self.fy, maximum)
            xs = np.repeat(self.fx, len(ys))
            return np.column_stack((xs, ys))
        else:
            return np.column_stack((x, self.calc_standard_form(x)))

    def to_numpy_array(self):
        """ Converts the parabola to a 1d array """
        return np.array([
            self.fx, self.fy,
            self.va, self.vh, self.vk,
            self.sa, self.sb, self.sc
            ])

    def to_array(self):
        """ Utility method to call to_numpy_array """
        return self.to_numpy_array()

    def __eq__(self, parabola2):
        """ Compare two parabolas """
        assert(isinstance(parabola2, Parabola))
        if parabola2 is None:
            return False
        a = self.to_numpy_array()
        b = parabola2.to_numpy_array()
        return np.allclose(a, b)

    def get_focus(self):
        """ Get the Focus point of this parabola. 2d np.array """
        return np.array([self.fx, self.fy])

    @staticmethod
    def to_standard_form(a, h, k):
        """ Calculate the standard form of the parabola from a vertex form """
        raise DeprecationWarning("Use Vert/Standard Form dataclass")

    @staticmethod
    def to_vertex_form(a, b, c):
        """ Calculate the vertex form of the parabola from a standard form """
        raise DeprecationWarning("Use Vert/Standard Form dataclass")
