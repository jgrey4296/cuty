"""
Provides the Parabola Class, mainly used for Voronoi calculation
"""
import logging as root_logger
import numpy as np
from .quadratic import Quadratic as Q

logging = root_logger.getLogger(__name__)
MAX = 5000

class Parabola:
    """ A class to represent and calculate a parabola. holds both forms of definition,
    focus/directrix AND standard form.
    Handles the degenerate case of focus and directrix being the same by designating
    as a vertical line
    """
    id = 0

    def __init__(self, fx, fy, d):
        """ Create a parabola with a focus x and y, and a directrix d """
        #pylint: disable=invalid-name
        self.id = Parabola.id
        Parabola.id += 1
        self.vertical_line = True
        self.fx = fx
        self.fy = fy
        self.d = d
        #focal parameter: the distance from vertex to focus/directrix
        self.p = 0.5 * (self.fy - self.d)
        #Vertex form: y = a(x-h)^2 + k
        if np.allclose(self.fy, self.d):
            self.va = 0
        else:
            self.va = 1/(2*(self.fy-self.d))
        self.vh = -self.fx
        self.vk = self.fy - self.p
        #standard form: y = ax^2 + bx + c
        self.sa = self.va
        self.sb = 2 * self.sa * self.vh
        self.sc = self.sa * (pow(self.vh, 2)) + self.vk

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return "y = {0:.2f} * x^2 + {1:.2f} x + {2:.2f}".format(self.sa, self.sb, self.sc)

    def __repr__(self):
        return "y = {} * x^2 + {} x + {}".format(self.sa, self.sb, self.sc)

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
        return (self.sa * pow(x, 2)) + (self.sb * x) + self.sc

    def calc_vertex_form(self, x):
        """ Get the y value of the parabola at an x position using
            the vertex form equation. Should equal calc_standard_form(x)
        """
        return self.va * pow(x + self.vh, 2) + self.vk

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
        return [
            a,
            -2*a*h,
            a*pow(h, 2)+k
        ]

    @staticmethod
    def to_vertex_form(a, b, c):
        """ Calculate the vertex form of the parabola from a standard form """
        return [
            a,
            -b/(2*a),
            c-(a * (pow(b, 2) / 4 * a))
        ]
