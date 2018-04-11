""" Line: Representation of a 2d line """
from math import sqrt
from numbers import Number
import logging as root_logger
import numpy as np
from ..math import intersect, get_distance
import IPython

logging = root_logger.getLogger(__name__)

CENTRE = np.array([[0.5, 0.5]])

class Line:
    """ A line as a start x and y, a direction, and a length, useful for
    algebraic manipulation """

    def __init__(self, s, d, l, swapped=False):
        assert(all([isinstance(x, np.ndarray) for x in [s,d]]))
        self.source = s
        self.direction = d
        self.length = l
        self.swapped = swapped

    def __repr__(self):
        return "Line(S: {}, D: {}, L: {}, SW: {})".format(self.source,
                                                          self.direction,
                                                          self.length,
                                                          self.swapped)
        
    def constrain(self, min_x, min_y, max_x, max_y):
        """ Intersect the line with a bounding box, adjusting points as necessary """
        #min and max: [x,y]
        dest = self.destination()
        nline = np.row_stack((self.source, dest))

        xs_min = nline[:,0] < min_x
        xs_max = nline[:,0] > max_x
        ys_min = nline[:,1] < min_y
        ys_max = nline[:,1] > max_y
        xs = (nline[:,0] * np.invert(xs_min + xs_max)) + (min_x * xs_min) + (max_x * xs_max)
        ys = (nline[:,1] * np.invert(ys_min + ys_max)) + (min_y * ys_min) + (max_y * ys_max)
        return np.column_stack((xs,ys))
        

    def destination(self):
        """ Calculate the destination vector of the line """
        return self.source + (self.length * self.direction)
        # ex = self.source[0] + (self.length * self.direction[0])
        # ey = self.source[1] + (self.length * self.direction[1])
        # return np.array([ex, ey])

    def bounds(self):
        if self.swapped:
            return np.row_stack((self.destination(), self.source))
        else:
            return np.row_stack((self.source, self.destination()))

    @staticmethod
    def newLine(a):
        """ Create a new line from two vertices """
        assert(isinstance(a, np.ndarray))
        assert(a.shape == (2,2))
        #Calculate the line parameters:
        vec = a[1] - a[0]
        l = sqrt(pow(vec, 2).sum())
        scale = 0
        if l != 0:
            scale = 1/l
        d = vec * scale
        #cx = a.x + (dx * l)
        #cy = a.y + (dy * l)
        return Line(a[0], d, l)


    def intersect_with_circle(self, centre, radius):
        #from http://csharphelper.com/blog/2014/09/determine-where-a-line-intersects-a-circle-in-c/
        A = sum(pow(self.direction,2))
        B =  2 * sum(self.direction * (self.source - centre))
        C = sum(pow(self.source - centre, 2)) - pow(radius, 2)

        det = pow(B,2) - (4 * A * C)

        if np.isclose(A, 0):
            raise Exception("No Intersection")
        
        if np.isclose(det, 0):
            t = -B / (2 * A)
            result =  np.array([self.source + (t * self.direction)])
            return result

        #two intersections:
        t = (-B + sqrt(det)) / (2 * A)
        t2 = (-B - sqrt(det)) / (2 * A)

        result = self.source + (np.array([[t],[t2]]) * self.direction)
        return result

    def __lt__(self, other):
        #TODO
        return False

    def __gt__(self, other):
        #TODO
        return False

    def intersect(self, other):
        #TODO
        return False

    
        
