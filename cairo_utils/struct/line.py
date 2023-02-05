""" Line: Representation of a 2d line """
##-- imports
from __future__ import annotations
import logging as root_logger
from dataclasses import InitVar, dataclass, field
from math import sqrt
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

import numpy as np

##-- end imports

from typing import Final

logging = root_logger.getLogger(__name__)

CENTRE : Final = np.array([[0.5, 0.5]])

@dataclass
class Line:
    """
    A line as a start position, a unit direction, and a length,
    useful for algebraic manipulation
    """

    source               : np.ndarray = field()
    direction            : np.ndarray = field()
    length               : float      = field()
    #slope and intersect :
    slope                : float      = field()
    slope_intersect      : np.ndarray = field()
    swapped              : bool       = field(default=False)
    origin_array         : np.ndarray = field(default=None)

    @staticmethod
    def build(a):
        """
        Create a new line from two vertices
        """
        assert(isinstance(a, np.ndarray))
        assert(a.shape == (2, 2))
        #Calculate the line parameters:
        vec = a[1] - a[0]
        l = sqrt(pow(vec, 2).sum())
        scale = 0
        if l != 0:
            scale = 1/l
        d = vec * scale
        #cx = a.x + (dx * l)
        #Slope and intersect:
        q = a[1] - a[0]
        if q[0] == 0:
            m = None
            b = None
        else:
            m = q[1] / q[0]
            b = a[0, 1] - (m * a[0, 0])
        return Line(a[0], d, l, m, b, origin_array=a)

    def __post_init__(self):
        assert(all([isinstance(x, np.ndarray) for x in [s, d]]))

    def __repr__(self):
        return "<Line S: {}, D: {}, L: {}, SW: {}>".format(self.source,
                                                           self.direction,
                                                           self.length,
                                                           self.swapped)

    def __call__(self, x=None, y=None):
        """
        Solve the line for the None value.
        must have either x or y passed in
        """
        assert(any([a is not None for a in [x, y]]))
        assert(not all([a is not None for a in [x, y]]))
        if x is not None:
            if self.slope is not None:
                yprime = self.slope * x + self.slope_intersect
            else:
                yprime = self.source[1]
            return np.array([x, yprime])
        else:
            if self.slope is not None and self.slope != 0:
                xprime = (y - self.slope_intersect) / self.slope
            else:
                xprime = self.source[0]
            return np.array([xprime, y])

    def constrain(self, min_x, min_y, max_x, max_y):
        """
        Intersect the line with a bounding box, adjusting points as necessary
        """
        #min and max: [x, y]
        dest = self.destination()
        nline = np.row_stack((self.source, dest))

        xs_min = nline[:, 0] < min_x
        xs_max = nline[:, 0] > max_x
        ys_min = nline[:, 1] < min_y
        ys_max = nline[:, 1] > max_y
        xs = (nline[:, 0] * np.invert(xs_min + xs_max)) + (min_x * xs_min) + (max_x * xs_max)
        ys = (nline[:, 1] * np.invert(ys_min + ys_max)) + (min_y * ys_min) + (max_y * ys_max)
        return np.column_stack((xs, ys))

    def subdivide(self, n):
        """
        Subdivide this line into n subdivisions
        """
        assert(isinstance(n, int))
        #plus two
        subdivisions = np.linspace(0, 1, n+2).reshape((-1, 1))
        new_points = self.source + ((subdivisions * self.length) * self.direction)
        return new_points

    def ratio_subdivide(self, ss, srange=None):
        """
        subdivides a line by a set of ratios that sum to 1
        """
        assert(isinstance(ss, np.ndarray))
        assert(0.99 <= ss.sum() < 1.1)
        if srange is None:
            srange = (0, 1)
        assert(isinstance(srange, tuple))
        assert(srange[0] <= srange[1])

        new_points = np.array([self.source])
        current_r = 0.0
        ratios = list(ss[0])
        while bool(ratios):
            r = ratios.pop(0)
            if srange[1] < r:
                ratios.append(r - srange[1])
                r = srange[1]
            current_r += r
            if bool(ratios) and r <= srange[0]:
                continue
            if not bool(ratios) and current_r != 1:
                current_r = 1
            new_points = np.row_stack((new_points,
                                       self.source + ((current_r * self.length) * self.direction)))

        return new_points

    def destination(self, l=None, r=None):
        """
        Calculate the destination vector of the line
        """
        if r is not None:
            l = self.length * r
        if l is None:
            l = self.length
        return self.source + (l * self.direction)

    def bounds(self):
        """
        Get the start and end points of this line
        """
        if self.swapped:
            return np.row_stack((self.destination(), self.source))
        else:
            return np.row_stack((self.source, self.destination()))

    def intersect_with_circle(self, centre, radius):
        """
        Get the point that this line crosses a circle
        from http://csharphelper.com/blog/2014/09/determine-where-a-line-intersects-a-circle-in-c/
        """
        #pylint: disable=invalid-name
        A = sum(pow(self.direction, 2))
        B = 2 * sum(self.direction * (self.source - centre))
        C = sum(pow(self.source - centre, 2)) - pow(radius, 2)

        det = pow(B, 2) - (4 * A * C)

        if np.isclose(A, 0):
            raise Exception("No Intersection")

        if np.isclose(det, 0):
            t = -B / (2 * A)
            result = np.array([self.source + (t * self.direction)])
            return result

        #two intersections:
        t = (-B + sqrt(det)) / (2 * A)
        t2 = (-B - sqrt(det)) / (2 * A)

        result = self.source + (np.array([[t], [t2]]) * self.direction)
        return result

    def intersect(self, other):
        """ Intersect a line with another """
        #TODO implement line intersection
        return False
