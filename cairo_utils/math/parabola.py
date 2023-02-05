"""
Provides the Parabola Class, mainly used for Voronoi calculation
"""
##-- imports
import logging as root_logger
from dataclasses import InitVar, dataclass, field
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

import numpy as np

from .quadratic import Quadratic as Q

##-- end imports

from typing import Final
import cairo_utils as cu

logging     = root_logger.getLogger(__name__)

MAX : Final = 5000

class Parabola:
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

    def calc(self, x):
        """ For given xs, return an (n, 2) array of xy pairs of the parabola """
        return np.column_stack((x, self.calc_vertex_form(x)))

    def to_np_array(self):
        """ Converts the parabola to a 1d array """
        return np.array([
            self.fx, self.fy,
            self.va, self.vh, self.vk,
            self.sa, self.sb, self.sc
            ])

    def get_focus(self):
        """ Get the Focus point of this parabola. 2d np.array """
        return np.array([self.fx, self.fy])
